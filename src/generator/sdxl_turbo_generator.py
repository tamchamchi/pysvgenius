from base import ITextToImageGenerator
from typing import List

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image


class SDXLTurboGenerator(ITextToImageGenerator):
    def __init__(
        self,
        model_path: str = "stabilityai/sdxl-turbo",
        guidance_scale: float = 0.0,
        num_inference_steps: int = 4,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.model_path = model_path
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.seed = seed
        self.pipe = self._load_pipeline()

    def _load_pipeline(self):
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_path, torch_dtype=torch.float16, variant="fp16"
            ).to(self.device)
            return pipe
        except Exception as e:
            raise ValueError(f"Failed to load pipeline: {e}") from e

    def process(
        self,
        prompt,
        num_images: int = 3,
        negative_prompt: str = "",
        num_inference_steps: int = 4,
        guidance_scale: float = 0,
        height: int = 512,
        width: int = 512,
        **kwargs,
    ) -> List[Image.Image]:
        try:
            generator = torch.Generator("cuda")
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
            )
            return outputs.images
        except Exception as e:
            raise RuntimeError(f"Error in process(): {e}") from e


if __name__ == "__main__":
    generator = SDXLTurboGenerator()
    images = generator.process("a big dog")
    print(len(images))
