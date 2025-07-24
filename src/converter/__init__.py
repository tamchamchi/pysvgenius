# src/converter/__init__.py
"""
Converter module for image-to-SVG conversion.

This module provides image-to-SVG conversion capabilities using various algorithms
like VTracer. It includes a factory pattern for easy instantiation of different
converter implementations.
"""

from .base import IConverter
from .vtracer import VtracerConverter
from .vtracer_v2 import VtracerConverterV2

__all__ = [
    "IConverter",
    "VtracerConverter",
    "VtracerConverterV2"
]
