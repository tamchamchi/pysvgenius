# src/converter/__init__.py
"""
Converter module for image-to-SVG conversion.

This module provides image-to-SVG conversion capabilities using various algorithms
like VTracer. It includes a factory pattern for easy instantiation of different
converter implementations.
"""

from .base import IConverter
from .factory import ConverterFactory
from .vtracer import VtracerConverter

# Auto-register available converters
ConverterFactory.register("vtracer", VtracerConverter)
ConverterFactory.register("vt", VtracerConverter)  # Short name

__all__ = [
    "IConverter",
    "ConverterFactory",
    "VtracerConverter"
]
