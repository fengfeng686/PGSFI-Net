# -*- coding: utf-8 -*-
"""
Inference performance benchmark for PGSFINet.

Measures, on a single full image (DRIVE 565x584 by default):
  1. Full-image latency  : end-to-end time of patch extraction + all-patch inference + patch stitching
  2. Throughput          : patches/sec and images/sec
  3. Peak memory         : peak GPU memory during single-patch (batch=1) inference

Usage:
  python benchmark_inference.py
  python benchmark_inference.py --img_h 565 --img_w 584 --stride 5 --warmup 10 --repeat 20
  python benchmark_inference.py --weight ./log/experiments/DRIVE/checkpoint_epoch_040.pth
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, './utils/')
sys.path.insert(0, './models/')
from models import MODELS
from utils.extract_patches import (
    paint_border_overlap,
    extract_ordered_overlap,
    recompone_overlap,
)


def parse_args():
    p = argparse.ArgumentParser(description='Inference benchmark for PGSFINet')
    p.add_argument('--model', default='pgsfi_net')
    p.add_argument('--weight', default='./log/experiments/DRIVE/checkpoint_epoch_011.pth',
                   help='checkpoint path')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--img_h', type=int, default=565, help='full image height')
    p.add_argument('--img_w', type=int, default=584, help='full image width')
    p.add_argument('--patch', type=int, default=48, help='patch size')
    p.add_argument('--stride', type=int, default=5, help='sliding window stride')
    p.add_argument('--batch', type=int, default=1, help='inference batch size (patches)')
    p.add_argument('--warmup', type=int, default=10, help='warmup forward passes')
    p.add_argument('--repeat', type=int, default=20, help='repetitions for single-patch latency')
    p.add_argument('--full_repeat', type=int, default=3,
                   help='repetitions for full-image latency (fewer because it is slow)')
    return p.parse_args()


def load_model(name, weight, gpu):
    model = MODELS[name](n_channels=1, n_classes=1)
    torch.cuda.set_device(gpu)
    checkpoint = torch.load(weight, map_location='cuda:{}'.format(gpu))
    state_dict = {k: v for k, v in checkpoint['state_dict'].items()
                  if not k.endswith('.total_ops') and not k.endswith('.total_params')}
    model.load_state_dict(state_dict, strict=False)
    model.cuda().eval()
    return model


def main():
    args = parse_args()

    print('=' * 64)
    print('Inference benchmark: {}'.format(args.model))
    print('weight  : {}'.format(args.weight))
    print('full img: {}x{} (HxW), patch {}x{}, stride {}'.format(
        args.img_h, args.img_w, args.patch, args.patch, args.stride))
    print('=' * 64)

    model = load_model(args.model, args.weight, args.gpu)
    n_params = sum(p.numel() for p in model.parameters())
    print('Model params: {:.2f} M'.format(n_params / 1e6))

    # ---- FLOPs (on a single patch, same size as actual inference) ----
    try:
        from thop import profile
        from copy import deepcopy
        model_copy = deepcopy(model).cpu()
        dummy_input = torch.randn(1, 1, args.patch, args.patch)
        flops, _ = profile(model_copy, inputs=(dummy_input,), verbose=False)
        del model_copy
        print('Model FLOPs ({}x{}): {:.2f} G'.format(args.patch, args.patch, flops / 1e9))
    except Exception as e:
        print('FLOPs: skipped ({})'.format(e))

    # ---- synthetic 1-channel full image (grayscale, [0,1]) ----
    # content-independent: latency/memory only depend on tensor shapes
    full_img = np.random.rand(1, 1, args.img_h, args.img_w).astype(np.float32)

    # ---- patch extraction (CPU, mirrors the real prediction pipeline) ----
    t0 = time.time()
    padded = paint_border_overlap(full_img, args.patch, args.patch, args.stride, args.stride)
    patches = extract_ordered_overlap(padded, args.patch, args.patch, args.stride, args.stride)
    t_extract = time.time() - t0

    n_patches = patches.shape[0]
    padded_h, padded_w = padded.shape[2], padded.shape[3]
    print('padded img: {}x{}, patches per image: {}'.format(padded_h, padded_w, n_patches))
    print('patch extraction (CPU) time: {:.3f} s'.format(t_extract))

    # ---- single-patch latency (batch=1 or --batch) ----
    patch_tensor = torch.from_numpy(patches[:args.batch]).float().cuda(args.gpu)

    # warmup (first CUDA kernels are slow)
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(patch_tensor)
    torch.cuda.synchronize()

    # reset peak memory stats, then measure one forward
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        t0 = time.time()
        out = model(patch_tensor)
        torch.cuda.synchronize()
        single_forward_time = time.time() - t0
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

    # latency over repeats
    latencies = []
    with torch.no_grad():
        for _ in range(args.repeat):
            torch.cuda.synchronize()
            t0 = time.time()
            _ = model(patch_tensor)
            torch.cuda.synchronize()
            latencies.append(time.time() - t0)
    latencies = np.array(latencies)
    mean_lat = float(latencies.mean())
    median_lat = float(np.median(latencies))

    # ---- full-image latency: extract (done above) + all patches + stitching ----
    # re-run patch inference over all patches, batch=1
    full_times = []
    for _ in range(args.full_repeat):
        preds = []
        with torch.no_grad():
            t_infer0 = time.time()
            for i in range(0, n_patches, args.batch):
                b = torch.from_numpy(patches[i:i + args.batch]).float().cuda(args.gpu)
                o = model(b).cpu().numpy()
                preds.append(o)
        torch.cuda.synchronize()
        t_infer1 = time.time()

        preds = np.concatenate(preds, axis=0)
        t_stitch0 = time.time()
        _ = recompone_overlap(preds, padded_h, padded_w, args.stride, args.stride)
        t_stitch1 = time.time()

        full_times.append((t_infer1 - t_infer0, t_stitch1 - t_stitch0))

    infer_times = np.array([x[0] for x in full_times])
    stitch_times = np.array([x[1] for x in full_times])
    mean_infer = float(infer_times.mean())
    mean_stitch = float(stitch_times.mean())
    mean_full = mean_infer + mean_stitch + t_extract

    print('-' * 64)
    print('Latency')
    print('  single-patch forward (batch={}): mean {:.3f} ms, median {:.3f} ms'.format(
        args.batch, mean_lat * 1e3, median_lat * 1e3))
    print('  full-image end-to-end          : {:.3f} s'.format(mean_full))
    print('    - patch inference ({} patches): {:.3f} s'.format(n_patches, mean_infer))
    print('    - patch stitching             : {:.3f} s'.format(mean_stitch))
    print('    - patch extraction            : {:.3f} s'.format(t_extract))

    print('-' * 64)
    print('Throughput')
    print('  patches/sec (batch={}) : {:.1f}'.format(args.batch, args.batch / mean_lat))
    print('  images/sec             : {:.4f}'.format(1.0 / mean_full))

    print('-' * 64)
    print('Peak memory (batch={} inference)'.format(args.batch))
    print('  peak allocated : {:.1f} MiB'.format(peak_alloc))
    print('  peak reserved  : {:.1f} MiB'.format(peak_reserved))
    print('=' * 64)


if __name__ == '__main__':
    main()
