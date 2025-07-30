from .base import IGenerator
from ..common import registry


@registry.register_generator("stable-diffusion-v2")
class SDv2Generator(IGenerator):
    def __init__(self):
        pass

    def process(self, prompt, num_images=3, **kwargs):
        pass
