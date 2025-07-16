# src/converter/__init__.py
"""
Converter module for image-to-SVG conversion.

This module provides image-to-SVG conversion capabilities using various algorithms
like VTracer. It includes a factory pattern for easy instantiation of different
converter implementations.
"""

from .base import IImageToConverter
from .factory import ImageToConverterFactory
from .vtracer import VtracerConverter

# Auto-register available converters
ImageToConverterFactory.register("vtracer", VtracerConverter)
ImageToConverterFactory.register("vt", VtracerConverter)  # Short name

__all__ = [
    "IImageToConverter",
    "ImageToConverterFactory",
    "VtracerConverter"
]
