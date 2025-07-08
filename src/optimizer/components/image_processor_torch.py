import kornia.filters as K
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diff_jpeg import diff_jpeg_coding
from kornia.filters.kernels import get_binary_kernel2d
from PIL import Image
from torch import fft, nn
from torchvision.transforms import InterpolationMode


class ImageProcessorTorch(nn.Module):
    def __init__(self, seed=None):
        super.__init__()
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def apply_fft_low_pass(
        self, images: torch.Tensor, cutoff_frequency: float = 0.5
    ) -> torch.Tensor:
        pass

    def apply_random_crop_resize(self, crop_percent: float = 0.05) -> torch.Tensor:
        pass

    def apply_median_filter(self, images: torch.Tensor, size: int = 3) -> torch.Tensor:
        pass

    def apply_bilateral_filter(
        self, images: torch.Tensor, d=9, sigma_color=75, sigma_space=75
    ) -> torch.Tensor:
        pass

    def apply_jpeg_compression(self, images: torch.Tensor, quality=85) -> torch.Tensor:
        pass

    def forward(
        self,
        x: torch.Tensor,
        skip_fft_low_pass=False,
        skip_random_crop_resize=False,
        skip_median_filter=False,
        skip_bilateral_filter=False,
        skip_jpeg_compression=False,
    ) -> torch.Tensor:
        pass
