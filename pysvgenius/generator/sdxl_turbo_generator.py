from typing import List, Optional

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

from ..common.registry import registry
from .base import IGenerator


@registry.register_generator("sdxl-turbo")
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
    """

    def __init__(
        self,
        model_path: str = "stabilityai/sdxl-turbo",
        device: str = "cuda",
        seed: int = 42,
        prefix: str = "flat color illustration, app icon,",
        suffix: str = ",inspired by Tom Whalen, atmospheric light, soft color palette, bold outlines, golden hour lighting.",
        lora_path: Optional[str] = None,
    ):
        """
        Initialize the SDXLTurboGenerator with model path, device, seed, and LoRA path.

        Args:
            model_path (str): Hugging Face model ID or local path.
            device (str): Target device (e.g., "cuda", "cpu").
            seed (int): Random seed for reproducibility.
            lora_path (Optional[str]): Optional path to a LoRA adapter.
        """
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device
        self.prefix = prefix
        self.suffix = suffix
        self.seed = seed

        # Load diffusion pipeline into memory
        self.pipe = self._load_pipeline()

    def _build_prompt(self, desc: str) -> str:
        return f"{self.prefix} {desc} {self.suffix}"

    def _load_pipeline(self):
        """
        Load the text-to-image generation pipeline from the pretrained model.

        Returns:
            A diffusers AutoPipelineForText2Image object.

        Raises:
            ValueError: If loading the pipeline fails.
        """

        try:
            # Load model using FP16 precision for faster inference
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_path, torch_dtype=torch.float16, variant="fp16"
            ).to(self.device)

            # Load LoRA if provided
            if self.lora_path:
                try:
                    pipe.load_lora_weights(self.lora_path)
                except Exception as e:
                    raise ValueError(
                        f"Failed to load LoRA weights: {e}") from e

            return pipe
        except Exception as e:
            raise ValueError(f"Failed to load pipeline: {e}") from e

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
        try:
            # Create a deterministic torch.Generator with fixed seed
            generator = torch.Generator(self.device)
            if self.seed is not None:
                generator.manual_seed(self.seed)

            processed_prompt = self._build_prompt(prompt)

            # Run the text-to-image pipeline
            outputs = self.pipe(
                prompt=processed_prompt,
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
            return outputs.images

        except Exception as e:
            raise RuntimeError(f"Error in process(): {e}") from e

    @classmethod
    def from_config(cls, cfg=None):
        model_path = cfg.get("model_path", "stabilityai/sdxl-turbo")
        device = cfg.get("device", "cpu")
        seed = cfg.get("seed", 42)
        prefix = cfg.get("prefix", "")
        suffix = cfg.get("suffix", "")
        lora_path = cfg.get("lora", None)
        return cls(model_path=model_path, device=device, seed=seed, prefix=prefix, suffix=suffix, lora_path=lora_path)


if __name__ == "__main__":
    # Example usage
    generator = SDXLTurboGenerator()
    images = generator.process("a big dog running in the field")
    # images = generator("a pick dog", num_images=3)
    print(f"Generated {len(images)} image(s)")
