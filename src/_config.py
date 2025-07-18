"""
_config.py

This module provides a global configuration handler for the project.
It supports loading configuration from a YAML file and accessing/modifying
config values at runtime.

Typical use cases include setting project directories, model parameters,
and runtime flags.
"""
import torch

import os

# Fixed project directories
PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJ_DIR, "models")
DATA_DIR = os.path.join(PROJ_DIR, "data")

AESTHETIC_MODEL_PATH = os.path.join(
    MODEL_DIR, "sac+logos+ava1-l14-linearMSE.pth")
CLIP_MODEL_PATH = os.path.join(MODEL_DIR, "ViT-L-14.pt")

DEFAULT_IMAGE_SIZE = 384
DEFAULT_FFT_CUTOFF = 0.5
DEFAULT_CROP_PERCENT = 0.05
FORWARD_CROP_PERCENT = 0.03
FORWARD_JPEG_QUALITY_1 = 95
FORWARD_MEDIAN_SIZE = 9
FORWARD_FFT_CUTOFF = 0.5
FORWARD_BILATERAL_D = 5
FORWARD_BILATERAL_SIGMA_COLOR = 75
FORWARD_BILATERAL_SIGMA_SPACE = 75
FORWARD_JPEG_QUALITY_2 = 92
TEST_JPEG_QUALITY = 85
COMPARISON_ATOL = 1e-2

class OptimizationArgs:
    iterations = 100
    jpeg_iter = 100
    aesthetic_iter = 0
    warmup_iter = 0
    log_interval = 20
    w_aesthetic = 200
    w_siglip = 100
    w_mse = 5000
    batch_size = 1
    lr_points = 0.1
    lr_color = 0.01
    grad_clip_norm = 1.0
    similarity_mode = ""

    device = "cuda"
    dtype = torch.float16