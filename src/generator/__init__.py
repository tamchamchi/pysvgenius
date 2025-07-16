# src/generator/__init__.py
"""
Generator module for text-to-image generation.

This module provides text-to-image generation capabilities using various models
like SDXL-Turbo. It includes a factory pattern for easy instantiation of different
generator implementations.
"""

from .base import ITextToImageGenerator
from .factory import TextToImageGeneratorFactory
from .sdxl_turbo_generator import SDXLTurboGenerator

# Auto-register available generators
TextToImageGeneratorFactory.register("sdxl-turbo", SDXLTurboGenerator)
TextToImageGeneratorFactory.register(
    "sdxl_turbo", SDXLTurboGenerator)  # Alternative name

__all__ = [
    "ITextToImageGenerator",
    "TextToImageGeneratorFactory",
    "SDXLTurboGenerator"
]
