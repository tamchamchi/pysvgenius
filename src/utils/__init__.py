# src/utils/__init__.py
"""
Utilities module for common functionality.

This module provides utility functions for image processing, SVG manipulation,
and logging functionality used across the pysvgenius library.
"""

from .logger import get_library_logger, create_console_logger, setup_logger
from .image_utils import prepare_image_for_ranking, svg_to_png, ImageProcessor, compare_pil_images
from .svg_utils import optimize_svg_with_scour, optimize_svg_size

__all__ = [
    # Logger utilities
    "get_library_logger",
    "create_console_logger",
    "setup_logger",

    # Image utilities
    "prepare_image_for_ranking",
    "svg_to_png",
    "ImageProcessor",
    "compare_pil_images",

    # SVG utilities
    "optimize_svg_with_scour",
    "optimize_svg_size"
]
