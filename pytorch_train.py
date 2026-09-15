import matplotlib
import time
import sys
from glob import glob
from os.path import abspath, dirname, isdir, isfile, join
from os import makedirs, fsync
from models import MODELS
from utils.extract_patches import get_data_training
import torch.nn as nn
import numpy as np
from torchsummary import summary
import torch.optim as optim
from sklearn.metrics import roc_auc_score, confusion_matrix

sys.path.insert(0, './utils/')
from utils.Data_loader import Retina_loader
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.cuda import empty_cache

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser = argparse.ArgumentParser(description="nasopharyngeal training")

parser.add_argument('--mode', default='gpu', type=str, metavar='train on gpu or cpu',
                    help='train on gpu or cpu(default gpu)')
parser.add_argument('--gpu', default=0, type=int, help='gpu number')
parser.add_argument('--optimizer', default='Adam',
                    choices=['Adam', 'SGD'],
                    help='loss: ' +
                         ' | '.join(['Adam', 'SGD']) +
                         ' (default: Adam)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--finetuning', type=str2bool, default=False,
                    help='is fine tuning')
parser.add_argument('--decay', type=float, default=1e-4,
                    help='decay of learning process')
parser.add_argument('--printfreq', type=int, default=1,
                    help='printfreq show training loss')
parser.add_argument('--itersize', type=int, default=100,
                    help='itersize of learning process')
parser.add_argument('--tensorboard-dir', default="tb",
                    help='name of the tensorboard data directory')
parser.add_argument('--checkpoint-interval', type=int, default=10,
                    help='checkpoint interval')
args = parser.parse_args()

gpuid = args.gpu
mode = args.mode
lr = args.lr
decay = args.decay
itersize = args.itersize
printfreq = args.printfreq
checkpoint_interval = args.checkpoint_interval
finetuning = args.finetuning


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def load_pretrained(model, fname, optimizer=None):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer, checkpoint['epoch']
        else:
            return model, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))


# ========= Load settings from Config file
config.read('configuration.txt')
algorithm = config.get('experiment name', 'name')
dataset = config.get('data attributes', 'dataset')
log_path_experiment = './log/experiments/' + algorithm + '/' + dataset + '/'
# log_path_experiment = './log/experiments/STARE/' + algorithm + '/'

# ========= Load settings from Config file
path_data = config.get('data paths', 'path_local')
model_path = config.get('data paths', 'model_path')
# training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
inp_shape = (int(config.get('data attributes', 'patch_width')), int(config.get('data attributes', 'patch_height')), 1)

THIS_DIR = abspath(dirname(log_path_experiment))
TMP_DIR = log_path_experiment
if not isdir(TMP_DIR):
    makedirs(TMP_DIR)
log = Logger(join(TMP_DIR, algorithm + '-log.txt'))
# log
sys.stdout = log
print('[i] Data name:            ', dataset)
print('[i] epochs:               ', N_epochs)
print('[i] Batch size:           ', batch_size)
print('[i] algoritm:             ', algorithm)
print('[i] gpu:                  ', args.gpu)
print('[i] mode:                 ', args.mode)
print('[i] learning rate:        ', args.lr)
print('[i] optimizer:            ', args.optimizer)
print('[i] finetuning:           ', finetuning)
fcn = True
if 'unet' not in algorithm:
    fcn = False
else:
    fcn = True
tensorboardPath = TMP_DIR + "/tensorboard"


def to_cuda(t, mode):
    if mode == 'gpu':
        return t.cuda()
    return t


def main():
    torch.manual_seed(0)

    # model = Gland_Edge()
    # model = RAUnet(input_channel=3, filter_num=8)

    # x = torch.randn(1, 3, 256, 256).requires_grad_(True)
    # y = model(x)
    # mvis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    # mvis.render(filename="raunet.jpg", directory=TMP_DIR)

    # model = UNet(n_channels=3, n_classes=1)
    if 'unet' not in algorithm:
        model = MODELS[algorithm](n_channels=1, n_classes=2)
    else:
        model = MODELS[algorithm](n_channels=1, n_classes=1)

    print('lr:' + str(lr))
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               weight_decay=decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              weight_decay=decay, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, verbose=True)
    # x = torch.randn(1, 1, inp_shape[0], inp_shape[1]).requires_grad_(True)
    # prediction = model(x)
    # mvis = make_dot(prediction, params=dict(list(model.named_parameters()) + [('image', x)]))
    # mvis.render(filename="model.jpg", directory=TMP_DIR)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # ==================== Params + FLOPs ====================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*50}")
    print(f"Model Params (Total):    {total_params / 1e6:.2f} M")
    print(f"Model Params (Trainable):{trainable_params / 1e6:.2f} M")
    
    try:
        from thop import profile
        from copy import deepcopy
        model_copy = deepcopy(model)
        model_copy.cpu()
        dummy_input = torch.randn(1, 1, 96, 96)
        flops, _ = profile(model_copy, inputs=(dummy_input,), verbose=False)
        del model_copy
        print(f"Model FLOPs (96x96):     {flops / 1e9:.2f} G")
    except Exception as e:
        print(f"FLOPs: skipped (thop error, training unaffected)")
        print(f"  Reason: {e}")
    print(f"{'='*50}\n")
    
    if mode == 'gpu':
        model.cuda()
    # =========================================================

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    print(model)
    if mode == 'gpu':
        torch.cuda.set_device(gpuid)
        torch.cuda.manual_seed(0)
        model.cuda()
        summary(model, input_size=(1, inp_shape[0], inp_shape[1]))

    summary_writer = SummaryWriter(tensorboardPath)

    patches_imgs_train, patches_masks_train = get_data_training(
        train_imgs_original=path_data + config.get('data paths', 'train_imgs_original'),
        train_groudTruth=path_data + config.get('data paths', 'train_groundTruth'),  # masks
        patch_height=inp_shape[0],
        patch_width=inp_shape[1],
        N_subimgs=int(config.get('training settings', 'N_subimgs')),
        inside_FOV=config.getboolean('training settings', 'inside_FOV'),
        fcn=fcn
    )
    # patches_imgs_test, patches_masks_test = get_data_training(
    #     train_imgs_original=path_data + config.get('data paths', 'test_imgs_original'),
    #     train_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
    #     patch_height=inp_shape[0],
    #     patch_width=inp_shape[1],
    #     N_subimgs=int(config.get('training settings', 'N_subimgs')),
    #     inside_FOV=config.getboolean('training settings', 'inside_FOV'),
    #     fcn=fcn
    # )
    # patches_imgs_train = np.concatenate([patches_imgs_train, patches_imgs_test], axis=0)
    # patches_masks_train = np.concatenate([patches_masks_train, patches_masks_test], axis=0)

    patches_imgs_train = np.transpose(patches_imgs_train, (0, 2, 3, 1))
    if fcn:
        patches_masks_train = np.transpose(patches_masks_train, (0, 2, 3, 1))
        train_dataset = Retina_loader(patches_imgs_train, patches_masks_train, 0.9, split='train')
        test_dataset = Retina_loader(patches_imgs_train, patches_masks_train, 0.9, split='test')
    else:
        train_dataset = Retina_loader(patches_imgs_train, patches_masks_train, 0.9, split='train', fcn=False)
        test_dataset = Retina_loader(patches_imgs_train, patches_masks_train, 0.9, split='test', fcn=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(N_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, epoch)
        save_file = join(TMP_DIR, 'checkpoint_epoch_%03d.pth' % (epoch + 1))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename=save_file)
        summary_writer.add_scalar(
            'train/loss', train_loss, epoch)
        summary_writer.add_scalar(
            'train/acc', train_acc, epoch)
        test_loss, test_acc, test_metrics = test(model, test_loader, epoch)
        scheduler.step(test_loss)
        summary_writer.add_scalar(
            'vali/loss', test_loss, epoch)
        summary_writer.add_scalar(
            'vali/acc', test_acc, epoch)
        
        if test_metrics is not None:
            summary_writer.add_scalar('vali/SE', test_metrics['SE'], epoch)
            summary_writer.add_scalar('vali/SP', test_metrics['SP'], epoch)
            summary_writer.add_scalar('vali/F1', test_metrics['F1'], epoch)
            summary_writer.add_scalar('vali/AUC', test_metrics['AUC'], epoch)
        
        log.flush()  # write log
    summary_writer.close()


def calculate_metrics(y_true, y_pred_scores, threshold=0.5):
    y_true = y_true.astype(int)
    y_pred = (y_pred_scores >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (1, 1):
        if y_true[0] == 1:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
    else:
        tn, fp, fn, tp = cm.ravel()
    
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    try:
        auc = roc_auc_score(y_true, y_pred_scores)
    except:
        auc = 0.0
    
    return {
        'SE': se,
        'SP': sp,
        'ACC': acc,
        'F1': f1,
        'AUC': auc,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }


def train(model, train_loader, optimizer, epoch):
    model.train()
    if mode == 'gpu':
        dtype_float = torch.cuda.FloatTensor
    else:
        dtype_float = torch.FloatTensor
    global net_vis
    end = time.time()
    pend = time.time()
    batch_time = Averagvalue()
    printfreq_time = Averagvalue()
    losses = Averagvalue()
    acc = Averagvalue()
    
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    for i, (image, label) in enumerate(train_loader):
        image = dtype_float(to_cuda(image.float(), mode)).requires_grad_(False)
        label = to_cuda(label, mode).requires_grad_(False)
        pre_label = model(image)
        if fcn:
            loss = BCELoss(pre_label, label)
            prec1 = accuracy_check(pre_label, label)
            acc.update(prec1, 1)
            
            all_preds.append(pre_label.cpu().detach().numpy().flatten())
            all_labels.append(label.cpu().detach().numpy().flatten())
        else:
            loss = CELoss(pre_label, label)
            prec1 = accuracy(pre_label, label)
            acc.update(prec1[0].item(), image.size(0))
        losses.update(loss.item(), image.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % (int(len(train_loader) / printfreq)) == 0:
            printfreq_time.update(time.time() - pend)
            pend = time.time()
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, N_epochs, i, len(train_loader)) + \
                   'printfreq time {printfreq_time.val:.3f} (avg:{printfreq_time.avg:.3f}) '.format(
                       printfreq_time=printfreq_time)
            print(info)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    empty_cache()
    
    if fcn and len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = calculate_metrics(all_labels, all_preds)
        print(f'TRAIN Metrics - SE: {metrics["SE"]:.4f}, SP: {metrics["SP"]:.4f}, ACC: {metrics["ACC"]:.4f}, F1: {metrics["F1"]:.4f}, AUC: {metrics["AUC"]:.4f}')
    
    return losses.avg, acc.avg


def test(model, test_loader, epoch):
    model.eval()
    epoch_time = Averagvalue()
    losses = Averagvalue()
    acc = Averagvalue()
    end = time.time()

    if mode == 'gpu':
        dtype_float = torch.cuda.FloatTensor
    else:
        dtype_float = torch.FloatTensor
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image = dtype_float(to_cuda(image.float(), mode)).requires_grad_(False)
            label = to_cuda(label, mode).requires_grad_(False)
            pre_label = model(image)
            if fcn:
                loss = BCELoss(pre_label, label)
                prec1 = accuracy_check(pre_label, label)
                acc.update(prec1, 1)
                
                all_preds.append(pre_label.cpu().detach().numpy().flatten())
                all_labels.append(label.cpu().detach().numpy().flatten())
            else:
                loss = CELoss(pre_label, label)
                prec1 = accuracy(pre_label, label)
                acc.update(prec1[0].item(), image.size(0))
            losses.update(loss.item(), image.size(0))
        empty_cache()
    
    epoch_time.update(time.time() - end)
    
    if fcn and len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = calculate_metrics(all_labels, all_preds)
        info = 'TEST Epoch: [{0}/{1}]'.format(epoch, N_epochs) + \
               'Test Epoch Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=epoch_time) + \
               'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses) + \
               f'SE: {metrics["SE"]:.4f}, SP: {metrics["SP"]:.4f}, ACC: {metrics["ACC"]:.4f}, F1: {metrics["F1"]:.4f}, AUC: {metrics["AUC"]:.4f}'
        print(info)
        
        return losses.avg, metrics['ACC'], metrics
    else:
        info = 'TEST Epoch: [{0}/{1}]'.format(epoch, N_epochs) + \
               'Test Epoch Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=epoch_time) + \
               'Acc {acc.val:f} (avg:{acc.avg:f}) '.format(acc=acc) + \
               'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(loss=losses)
        print(info)
        return losses.avg, acc.avg, None


def CELoss(y, label):
    loss = nn.CrossEntropyLoss()
    return loss(y, label)


def BCELoss(prediction, label):
    masks_probs_flat = prediction.view(-1)
    true_masks_flat = label.float().view(-1)
    loss = nn.BCELoss()(masks_probs_flat, true_masks_flat)
    return loss


def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.cpu().detach().numpy()
        np_ims.append(item)
    compare = np.equal(np.where(np_ims[0] > 0.5, 1, 0), np_ims[1])
    accuracy = np.sum(compare)
    return accuracy / len(np_ims[0].flatten())


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.div(batch_size))
    return res


if __name__ == '__main__':
    main()
