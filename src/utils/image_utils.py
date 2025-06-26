import numpy as np

from PIL import Image
from skimage.metrics import structural_similarity as ssim


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
    img1_gray = img1.resize(size, Image.Resampling.LANCZOS).convert('L')
    img2_gray = img2.resize(size, Image.Resampling.LANCZOS).convert('L')

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
