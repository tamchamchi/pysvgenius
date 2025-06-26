from typing import List, Optional

import torch
from base import ITextToImageGenerator
from diffusers import AutoPipelineForText2Image
from PIL import Image


class SDXLTurboGenerator(ITextToImageGenerator):
    """
    A text-to-image generator implementation using Stability AI's SDXL-Turbo model
    and Hugging Face's Diffusers library.

    This class provides an interface to generate images from text prompts with optional
    control over inference steps, guidance scale, resolution, seed, and LoRA fine-tuning.

    Attributes:
        model_path (str): Path or model ID for the pretrained SDXL-Turbo model.
        guidance_scale (float): Classifier-free guidance scale (usually 0.0 for SDXL-Turbo).
        num_inference_steps (int): Number of inference steps for image generation.
        device (str): Device to run the model on, typically "cuda" or "cpu".
        seed (int): Random seed for reproducible outputs.
        lora_path (Optional[str]): Path to a LoRA model if using LoRA fine-tuning.
    """

    def __init__(
        self,
        model_path: str = "stabilityai/sdxl-turbo",
        guidance_scale: float = 0.0,
        num_inference_steps: int = 4,
        device: str = "cuda",
        seed: int = 42,
        lora_path: Optional[str] = None,
    ):
        """
        Initialize the SDXLTurboGenerator with configuration parameters.

        Args:
            model_path (str): Hugging Face model ID or local path.
            guidance_scale (float): Guidance scale to steer image generation.
            num_inference_steps (int): Number of diffusion steps.
            device (str): Target device (e.g., "cuda", "cpu").
            seed (int): Random seed for reproducibility.
            lora_path (Optional[str]): Optional path to a LoRA adapter.
        """
        self.model_path = model_path
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.seed = seed
        self.lora_path = lora_path
        self.pipe = self._load_pipeline()

    def _load_pipeline(self):
        """
        Load the text-to-image generation pipeline from the pretrained model.

        Returns:
            A diffusers AutoPipelineForText2Image object.

        Raises:
            ValueError: If loading the pipeline fails.
        """
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_path, torch_dtype=torch.float16, variant="fp16"
            ).to(self.device)
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
            generator = torch.Generator(self.device)
            if self.seed is not None:
                generator.manual_seed(self.seed)

            outputs = self.pipe(
                prompt=prompt,
                num_images_per_prompt=num_images,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
            return outputs.images
        except Exception as e:
            raise RuntimeError(f"Error in process(): {e}") from e


if __name__ == "__main__":
    generator = SDXLTurboGenerator()
    images = generator.process("a big dog")
    print(len(images))
