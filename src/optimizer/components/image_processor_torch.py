import kornia.filters as K
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diff_jpeg import diff_jpeg_coding
from torch import fft, nn
from torchvision.transforms import InterpolationMode


class ImageProcessorTorch(nn.Module):
    """
    A differentiable image processor using PyTorch for frequency-based filtering.

    Args:
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    """
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

    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def apply_fft_low_pass(
        self, images: torch.Tensor, cutoff_frequency: float = DEFAULT_FFT_CUTOFF
    ) -> torch.Tensor:
        """
        Apply a low-pass filter in the frequency domain using FFT (Fast Fourier Transform).

        This method removes high-frequency details (edges, noise) from the image while preserving
        low-frequency components (smooth areas, large structures). Useful for smoothing or preparing
        images for aesthetic analysis.

        Args:
            images (torch.Tensor): A batch of input images with shape (B, C, H, W).
                                   Pixel values must be in [0, 1] and images must be square.
            cutoff_frequency (float, optional): Normalized cutoff frequency (0-1). Lower values
                                                remove more high-frequency content.
                                                Defaults to DEFAULT_FFT_CUTOFF.

        Returns:
            torch.Tensor: The filtered images with the same shape (B, C, H, W), clamped to [0, 1].
        """
        b, c, h, w = images.shape
        assert h == w, "The images must be square"
        rows, cols = h, w
        crow, ccol = rows // 2, cols // 2

        # FFT and center shift
        fshift = fft.fftshift(
            fft.fft2(images.to(dtype=torch.float32), dim=(-2, -1)), dim=(-2, -1)
        )

        # Scale cutoff to match current image resolution
        cutoff_frequency = cutoff_frequency * h / self.DEFAULT_IMAGE_SIZE
        r = min(crow, ccol) * cutoff_frequency
        r_sq = r**2

        # Create circular low-pass mask
        y, x = torch.meshgrid(
            torch.arange(rows, device=images.device),
            torch.arange(cols, device=images.device),
            indexing="ij",
        )
        center_dist_sq = (y - crow) ** 2 + (x - ccol) ** 2
        mask = (
            (center_dist_sq <= r_sq).to(torch.float16).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, H, W)

        # Apply mask and inverse FFT
        fshift_filtered = fshift * mask
        f_ishift = fft.ifftshift(fshift_filtered, dim=(-2, -1))
        img_back = fft.ifft2(f_ishift, dim=(-2, -1))
        img_back_real = torch.real(img_back)

        # Clamp values to [0, 1]
        filtered_images = torch.clamp(img_back_real, 0, 1)

        return filtered_images

    def apply_random_crop_resize(
        self,
        images: torch.Tensor,
        crop_percent: float = DEFAULT_CROP_PERCENT,
        interpolation=InterpolationMode.BILINEAR,
    ) -> torch.Tensor:
        """
        Apply random cropping followed by resizing to the original size for a batch of images.

        This is commonly used for data augmentation to improve generalization.

        Args:
            images (torch.Tensor): Batch of input images with shape (B, C, H, W), values in [0, 1].
            crop_percent (float): Max percentage of height/width to randomly crop (0-1).
            interpolation: Interpolation method for resizing. Default is bilinear.

        Returns:
            torch.Tensor: Batch of processed images with the same shape (B, C, H, W).
        """
        b, c, h, w = images.shape
        original_size = (h, w)

        crop_pixels_h = int(h * crop_percent)
        crop_pixels_w = int(w * crop_percent)

        cropped_images = []

        for i in range(b):
            left = self.rng.randint(0, crop_pixels_w + 1)
            top = self.rng.randint(0, crop_pixels_h + 1)
            right = w - self.rng.randint(0, crop_pixels_w + 1)
            bottom = h - self.rng.randint(0, crop_pixels_h + 1)
            new_w = right - left
            new_h = bottom - top
            # print(f"[PyTorch] left: {left}, top: {top}, right: {right}, bottom: {bottom}")

            # Crop & Resize (`TF.resized_crop` is differentiable)
            cropped_img = TF.resized_crop(
                images[i],
                top,
                left,
                new_h,
                new_w,
                original_size,
                interpolation=interpolation,
            )
            cropped_images.append(cropped_img)
        return torch.stack(cropped_images)

    def apply_median_filter(self, images: torch.Tensor, size: int = 3) -> torch.Tensor:
        """
        Apply a median filter to a batch of images using unfolding for efficiency.

        This method avoids using heavy convolutional operations and performs
        an exact median blur that is fully differentiable.

        Args:
            images (torch.Tensor): Input image tensor of shape (B, C, H, W), where
                                B = batch size, C = number of channels, H = height, W = width.
            size (int): Size of the median filter kernel (must be odd). Default is 3.

        Returns:
            torch.Tensor: Median-filtered image tensor of shape (B, C, H, W).
        """
        # Validate that kernel size is a positive integer
        if not isinstance(size, int) or size < 1:
            raise ValueError("kernel_size must be a positive integer.")

        # Ensure kernel size is odd (median needs a center)
        if size % 2 == 0:
            size += 1

        # Calculate padding size
        pad = (size - 1) // 2

        # Pad the image using replicate mode to handle borders
        x = F.pad(images, (pad, pad, pad, pad), mode="replicate")

        # Extract image patches of shape (B, C * size^2, H * W)
        patches = F.unfold(x, kernel_size=size)

        # Get dimensions
        B, CP, HW = patches.shape

        # Reshape to (B, C, size*size, H*W) to compute median across kernel window
        patches = patches.view(B, images.shape[1], size * size, HW)

        # Compute median along the kernel dimension (dim=2)
        med = torch.median(patches, dim=2).values

        # Reshape result to original image shape
        med = med.view_as(images)

        return med  # Return the median filtered images

    def apply_bilateral_filter(
        self, images: torch.Tensor, d=9, sigma_color=75, sigma_space=75
    ) -> torch.Tensor:
        """
        Apply a bilateral filter using Kornia (differentiable and GPU-friendly).

        Bilateral filtering smooths the image while preserving edges, by considering
        both spatial closeness and color similarity.

        Args:
            images (torch.Tensor): Input image tensor of shape (B, C, H, W),
                                with pixel values assumed to be in [0, 1].
            d (int, optional): Diameter of the filter kernel. Will be scaled based on image size. Defaults to 9.
            sigma_color (float, optional): Filter sigma in the color space (range 0–255). Controls how dissimilar colors are smoothed. Defaults to 75.
            sigma_space (float, optional): Filter sigma in the coordinate space. Controls how far pixels influence each other spatially. Defaults to 75.

        Returns:
            torch.Tensor: Bilateral-filtered image tensor of shape (B, C, H, W).
        """
        # Normalize sigma_color from 0–255 scale to [0, 1] for Kornia
        normalized_sigma_color = sigma_color / 255.0

        # Extract image shape
        b, c, h, w = images.shape

        # Check if the input is a square image (required by this implementation)
        assert h == w, "The images must be square"

        # Scale the kernel size d proportionally to the input size (relative to a default reference size)
        d = int(np.ceil(d * h / self.DEFAULT_IMAGE_SIZE))

        # Apply bilateral filter using Kornia
        return K.bilateral_blur(
            images,
            kernel_size=d,
            sigma_color=normalized_sigma_color,
            sigma_space=(sigma_space, sigma_space),  # horizontal and vertical
        )

    def apply_jpeg_compression(self, images: torch.Tensor, quality=85) -> torch.Tensor:
        """
        Simulate JPEG compression using diff_jpeg (optionally non-differentiable).

        Note:
            This function uses diff_jpeg to simulate the effect of lossy JPEG compression.
            When `ste=False`, the compression is not differentiable (i.e., gradients won't flow).

        Args:
            images (torch.Tensor): Input image tensor of shape (B, C, H, W),
                                with pixel values expected to be in the [0, 1] range.
            quality (int, optional): JPEG quality level (1–100); higher means better quality and less compression.
                                    Defaults to 85.

        Returns:
            torch.Tensor: JPEG-compressed (simulated) image tensor of shape (B, C, H, W),
                        with pixel values in the [0, 1] range.
        """
        # Get batch size and image dimensions
        b, c, h, w = images.shape

        # Create a tensor of JPEG quality values for each image in the batch
        quality_tensor = torch.tensor(
            # Repeat the quality value for each image
            [quality] * b,
            # Place it on the same device as the images (CPU/GPU)
            device=images.device,
            dtype=torch.float16,  # Use float16 for memory efficiency
        )

        # Perform JPEG compression simulation using diff_jpeg
        # Inputs are scaled to [0, 255] as JPEG encoding expects that range
        # `ste=False` disables straight-through estimator, making this step non-differentiable
        compressed_images = diff_jpeg_coding(
            images * 255.0, jpeg_quality=quality_tensor, ste=False
        )

        # Rescale the result back to [0, 1] before returning
        return compressed_images / 255.0

    def forward(
        self,
        x: torch.Tensor,
        skip_fft_low_pass=False,
        skip_random_crop_resize=False,
        skip_median_filter=False,
        skip_bilateral_filter=False,
        skip_jpeg_compression=False,
    ) -> torch.Tensor:
        """
        Apply a sequence of image degradation operations to the input tensor.

        This method simulates real-world image corruption by applying various
        differentiable and non-differentiable image transformations. These
        operations can help models learn more robust or aesthetically-aware
        representations. Each step can be skipped independently via its
        corresponding flag.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W) with pixel values in [0, 1].
            skip_fft_low_pass (bool): If True, skip FFT low-pass filtering.
            skip_random_crop_resize (bool): If True, skip random cropping and resizing.
            skip_median_filter (bool): If True, skip median filtering.
            skip_bilateral_filter (bool): If True, skip bilateral filtering.
            skip_jpeg_compression (bool): If True, skip JPEG compression (both passes).

        Returns:
            torch.Tensor: Output image tensor after transformations, same shape as input.
        """
        # 1. Apply random crop and resize (differentiable)
        if not skip_random_crop_resize:
            x = self.apply_random_crop_resize(
                x, crop_percent=self.FORWARD_CROP_PERCENT
            )

        # 2. Apply JPEG compression (1st pass - non-differentiable)
        if not skip_jpeg_compression:
            x = self.apply_jpeg_compression(
                x, quality=self.FORWARD_JPEG_QUALITY_1)

        # 3. Apply median filtering (partially differentiable)
        if not skip_median_filter:
            x = self.apply_median_filter(x, size=self.FORWARD_MEDIAN_SIZE)

        # 4. Apply FFT low-pass filter (differentiable)
        if not skip_fft_low_pass:
            x = self.apply_fft_low_pass(
                x, cutoff_frequency=self.FORWARD_FFT_CUTOFF)

        # 5. Apply bilateral filter (differentiable)
        if not skip_bilateral_filter:
            x = self.apply_bilateral_filter(
                x,
                d=self.FORWARD_BILATERAL_D,
                sigma_color=self.FORWARD_BILATERAL_SIGMA_COLOR,
                sigma_space=self.FORWARD_BILATERAL_SIGMA_SPACE,
            )

        # 6. Apply JPEG compression again (2nd pass - non-differentiable)
        if not skip_jpeg_compression:
            x = self.apply_jpeg_compression(
                x, quality=self.FORWARD_JPEG_QUALITY_2)

        return x

    def apply(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self(x, **kwargs)
