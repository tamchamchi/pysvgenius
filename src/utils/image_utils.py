import io

import cairosvg
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision import transforms


def compute_ssim_images(img1: Image.Image, img2: Image.Image, size=(384, 384)):
    """
    Compare two PIL images after resizing to a fixed size using SSIM.

    Parameters:
        img1 (PIL.Image.Image): The first image.
        img2 (PIL.Image.Image): The second image.
        size (tuple): Resize dimensions (default: 384x384).

    Returns:
        score (float): SSIM score.
    """
    # Resize and convert to grayscale
    img1_gray = img1.resize(size, Image.Resampling.LANCZOS).convert("L")
    img2_gray = img2.resize(size, Image.Resampling.LANCZOS).convert("L")

    # Convert to numpy arrays
    arr1 = np.array(img1_gray)
    arr2 = np.array(img2_gray)

    # Calculate SSIM
    score_ssim, _ = ssim(arr1, arr2, full=True)

    return score_ssim


def resize_image(image, size: tuple[int, int] = (384, 384)) -> Image.Image:
    """
    Resize a PIL image to the specified size using LANCZOS resampling.

    Parameters:
        image (PIL.Image.Image): The image to resize.
        size (tuple): The target size (width, height).

    Returns:
        PIL.Image.Image: The resized image.
    """
    resized_image = image.resize(size, Image.Resampling.LANCZOS)
    return resized_image


class ImageProcessor:
    def __init__(self, image: Image.Image, seed=None):
        """Initialize with either a path to an image or a PIL Image object."""
        self.image = image
        self.original_image = self.image.copy()
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

    def reset(self):
        self.image = self.original_image.copy()
        return self

    def visualize_comparison(
        self,
        original_name="Original",
        processed_name="Processed",
        figsize=(10, 5),
        show=True,
    ):
        """Display original and processed images side by side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.imshow(np.asarray(self.original_image))
        ax1.set_title(original_name)
        ax1.axis("off")

        ax2.imshow(np.asarray(self.image))
        ax2.set_title(processed_name)
        ax2.axis("off")

        title = f"{original_name} vs {processed_name}"
        fig.suptitle(title)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def apply_median_filter(self, size=3):
        """Apply median filter to remove outlier pixel values.

        Args:
             size: Size of the median filter window.
        """
        self.image = self.image.filter(ImageFilter.MedianFilter(size=size))
        return self

    def apply_bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filter to smooth while preserving edges.

        Args:
             d: Diameter of each pixel neighborhood
             sigma_color: Filter sigma in the color space
             sigma_space: Filter sigma in the coordinate space
        """
        # Convert PIL Image to numpy array for OpenCV
        img_array = np.asarray(self.image)

        # Apply bilateral filter
        filtered = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)

        # Convert back to PIL Image
        self.image = Image.fromarray(filtered)
        return self

    def apply_fft_low_pass(self, cutoff_frequency=0.5):
        """Apply low-pass filter in the frequency domain using FFT.

        Args:
             cutoff_frequency: Normalized cutoff frequency (0-1).
                  Lower values remove more high frequencies.
        """
        # Convert to numpy array, ensuring float32 for FFT
        img_array = np.array(self.image, dtype=np.float32)

        # Process each color channel separately
        result = np.zeros_like(img_array)
        for i in range(3):  # For RGB channels
            # Apply FFT
            f = np.fft.fft2(img_array[:, :, i])
            fshift = np.fft.fftshift(f)

            # Create a low-pass filter mask
            rows, cols = img_array[:, :, i].shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.float32)
            r = int(min(crow, ccol) * cutoff_frequency)
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
            mask[mask_area] = 1

            # Apply mask and inverse FFT
            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.real(img_back)

            result[:, :, i] = img_back

        # Clip to 0-255 range and convert to uint8 after processing all channels
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        self.image = Image.fromarray(result)
        return self

    def apply_jpeg_compression(self, quality=85):
        """Apply JPEG compression.

        Args:
             quality: JPEG quality (0-95). Lower values increase compression.
        """
        buffer = io.BytesIO()
        self.image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        self.image = Image.open(buffer)
        return self

    def apply_random_crop_resize(self, crop_percent=0.05):
        """Randomly crop and resize back to original dimensions.

        Args:
             crop_percent: Percentage of image to crop (0-0.4).
        """
        width, height = self.image.size
        crop_pixels_w = int(width * crop_percent)
        crop_pixels_h = int(height * crop_percent)

        left = self.rng.randint(0, crop_pixels_w + 1)
        top = self.rng.randint(0, crop_pixels_h + 1)
        right = width - self.rng.randint(0, crop_pixels_w + 1)
        bottom = height - self.rng.randint(0, crop_pixels_h + 1)

        self.image = self.image.crop((left, top, right, bottom))
        self.image = self.image.resize((width, height), Image.BILINEAR)
        return self

    def apply(self):
        """Apply an ensemble of defenses."""
        return (
            self.apply_random_crop_resize(crop_percent=0.03)
            .apply_jpeg_compression(quality=95)
            .apply_median_filter(size=9)
            .apply_fft_low_pass(cutoff_frequency=0.5)
            .apply_bilateral_filter(d=5, sigma_color=75, sigma_space=75)
            .apply_jpeg_compression(quality=92)
        )


def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """
    Converts an SVG string to a PNG image using CairoSVG.

    If the SVG does not define a `viewBox`, it will add one using the provided size.

    Parameters
    ----------
    svg_code : str
         The SVG string to convert.
    size : tuple[int, int], default=(384, 384)
         The desired size of the output PNG image (width, height).

    Returns
    -------
    PIL.Image.Image
         The generated PNG image.
    """
    # Ensure SVG has proper size attributes
    if "viewBox" not in svg_code:
        svg_code = svg_code.replace(
            "<svg", f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
    return Image.open(io.BytesIO(png_data)).convert("RGB").resize(size)


def process_svg_to_image(svg_code: str) -> Image.Image:
    image_processor = ImageProcessor(svg_to_png(svg_code), seed=42).apply()
    image = image_processor.image.copy()
    return image


def image_to_tensor(image: Image) -> torch.Tensor:
    to_tensor = transforms.ToTensor()

    if isinstance(image, list):
        tensor = torch.stack([to_tensor(img) for img in image])
    else:
        tensor = to_tensor(image)
    return tensor
