from typing import List

import torch
from .base import IGenerator
from diffusers import AutoPipelineForText2Image
from PIL import Image

from src.utils.logger import get_library_logger
from typing import Optional
import logging


class SDXLTurboGenerator(IGenerator):
    """
    A text-to-image generator implementation using Stability AI's SDXL-Turbo model
    and Hugging Face's Diffusers library.

    This class provides an interface to generate images from text prompts with optional
    control over inference steps, guidance scale, resolution, seed, and LoRA fine-tuning.

    Attributes:
        model_path (str): Path or model ID for the pretrained SDXL-Turbo model.
        device (str): Device to run the model on, typically "cuda" or "cpu".
        seed (int): Random seed for reproducible outputs.
        lora_path (Optional[str]): Path to a LoRA model if using LoRA fine-tuning.
    """

    def __init__(
        self,
        model_path: str = "stabilityai/sdxl-turbo",
        device: str = "cuda",
        seed: int = 42,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the SDXLTurboGenerator with model path, device, seed, and LoRA path.

        Args:
            model_path (str): Hugging Face model ID or local path.
            device (str): Target device (e.g., "cuda", "cpu").
            seed (int): Random seed for reproducibility.
            lora_path (Optional[str]): Optional path to a LoRA adapter.
        """
        if logger is not None:
            self.logger = logger
        else:
            self.logger = get_library_logger(
                f"{__name__}.{self.__class__.__name__}")
        self.model_path = model_path
        self.device = device
        self.seed = seed

        # Load diffusion pipeline into memory
        self.pipe = self._load_pipeline()
        self.logger.success("SDXLTurboGenerator initialization completed")

    def _load_pipeline(self):
        """
        Load the text-to-image generation pipeline from the pretrained model.

        Returns:
            A diffusers AutoPipelineForText2Image object.

        Raises:
            ValueError: If loading the pipeline fails.
        """
        self.logger.info(
            f"Loading SDXL-Turbo pipeline from: {self.model_path}")
        self.logger.debug(f"Target device: {self.device}")

        try:
            # Load model using FP16 precision for faster inference
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_path, torch_dtype=torch.float16, variant="fp16"
            ).to(self.device)

            self.logger.success(
                f"Successfully loaded SDXL-Turbo pipeline on {self.device}")
            return pipe
        except Exception as e:
            self.logger.error(f"Failed to load pipeline: {e}")
            raise ValueError(f"Failed to load pipeline: {e}") from e

    def set_params(self, **kwargs):
        """
        Update generator attributes such as seed, device, model_path, etc.

        Args:
            kwargs: Dictionary of parameter names and values.

        Raises:
            AttributeError: If a given parameter is not valid.
        """
        self.logger.debug(f"Updating parameters: {kwargs}")

        for key, val in kwargs.items():
            if hasattr(self, key):
                old_val = getattr(self, key)
                setattr(self, key, val)
                self.logger.debug(f"Updated {key}: {old_val} -> {val}")
            else:
                self.logger.error(
                    f"Invalid parameter '{key}' for {self.__class__.__name__}")
                raise AttributeError(
                    f"Parameter '{key}' is not valid for {self.__class__.__name__}"
                )

        self.logger.info("Parameters updated successfully")

    def get_params(self) -> dict:
        """
        Get current configuration parameters of the generator.

        Returns:
            dict: Dictionary containing model_path, device, seed, and lora_path.
        """
        return {
            "model_path": self.model_path,
            "device": self.device,
            "seed": self.seed,
        }

    def process(
        self,
        prompt: str,
        num_images: int = 3,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 0,
        height: int = 512,
        width: int = 512,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate one or more images from a text prompt using SDXL-Turbo.

        Args:
            prompt (str): The main text prompt for image generation.
            num_images (int): Number of images to generate per prompt.
            negative_prompt (str): Optional text describing what to avoid in the image.
            num_inference_steps (int): Number of denoising steps (default 4).
            guidance_scale (float): CFG scale (0.0 recommended for SDXL-Turbo).
            height (int): Output image height in pixels.
            width (int): Output image width in pixels.
            **kwargs: Additional arguments passed to the pipeline.

        Returns:
            List[Image.Image]: A list of generated PIL images.

        Raises:
            RuntimeError: If generation fails.
        """
        self.logger.info(
            f"Starting image generation with prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        self.logger.debug(f"Generation parameters: num_images={num_images}, steps={num_inference_steps}, "
                          f"guidance_scale={guidance_scale}, size={width}x{height}")

        try:
            # Create a deterministic torch.Generator with fixed seed
            generator = torch.Generator(self.device)
            if self.seed is not None:
                generator.manual_seed(self.seed)
                self.logger.debug(f"Using seed: {self.seed}")

            # Run the text-to-image pipeline
            self.logger.debug("Running SDXL-Turbo inference...")
            outputs = self.pipe(
                prompt=prompt,
                num_images_per_prompt=num_images,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs  # Forward any extra args
            )

            # Return list of generated PIL.Image objects
            self.logger.success(
                f"Successfully generated {len(outputs.images)} images")
            return outputs.images

        except Exception as e:
            self.logger.error(f"Error in process(): {e}")
            raise RuntimeError(f"Error in process(): {e}") from e


if __name__ == "__main__":
    # Example usage
    generator = SDXLTurboGenerator()
    images = generator.process("a big dog running in the field")
    images = generator("a pick dog", num_images=3)
    print(f"Generated {len(images)} image(s)")
