# PGSFI-Net

Official PyTorch implementation of **PGSFI-Net** for retinal vessel segmentation.

This repository provides the source code and related files used in our study, including the PGSFI-Net architecture, ablation variants, dataset preprocessing, model training, prediction, evaluation, and inference benchmarking scripts.

## Overview

PGSFI-Net is designed for pixel-level retinal vessel segmentation from color fundus images. The model integrates vessel-prior information with spatial- and frequency-domain feature processing to improve the representation of vessels with different widths, orientations, and local contrast levels.

The implementation uses image patches during training and overlapping patches during inference. By default, the model uses `48 x 48` patches and reconstructs full-resolution vessel segmentation maps from overlapping predictions.

## Repository Structure


PGSFI-Net/
├── configuration.txt              # Dataset paths and experiment settings
├── prepare_dataset.py             # Convert original datasets to HDF5 files
├── pytorch_train.py               # Model training
├── pytorch_predict_cnn.py         # Patch-based CNN prediction
├── pytorch_predict_fcn.py         # Overlapping-patch FCN prediction
├── evaluation.py                  # Segmentation performance evaluation
├── eval_cross_DRIVE.py            # Cross-dataset evaluation
├── benchmark_inference.py         # Inference latency and memory benchmark
├── benchmark_patch.py             # Patch-level benchmark
├── models/
│   ├── pgsfi_net.py               # Main PGSFI-Net model
│   ├── spatial_freq_domain.py     # Spatial-frequency modules
│   ├── unet.py                    # U-Net baseline
│   ├── deform_unet.py             # Deformable U-Net baseline
│   └── pgsfi_net_*.py             # Ablation and model variants
└── utils/
    ├── Data_loader.py             # PyTorch data loaders
    ├── extract_patches.py         # Patch extraction and reconstruction
    ├── pre_processing.py          # Image preprocessing
    ├── help_functions.py          # HDF5 and visualization utilities
    └── layers.py                  # Network layers

    
## Dataset

[DRIVE](http://www.isi.uu.nl/Research/Databases/DRIVE/), [STARE](http://cecas.clemson.edu/~ahoover/stare/), [CHASE_DB](https://blogs.kingston.ac.uk/retinal/chasedb1/)
