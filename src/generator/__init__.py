# src/generator/__init__.py
"""
Generator module for text-to-image generation.

This module provides text-to-image generation capabilities using various models
like SDXL-Turbo. It includes a factory pattern for easy instantiation of different
generator implementations.
"""

from .base import IGenerator
from .sdxl_turbo_generator import SDXLTurboGenerator
from .stable_diffusion_v2_generator import SDv2Generator


__all__ = [
    "IGenerator",
    "SDXLTurboGenerator",
    "SDv2Generator",
]
