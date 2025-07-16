# src/optimizer/__init__.py
"""
Optimizer module for SVG optimization using DiffVG.

This module provides SVG optimization capabilities using differentiable
vector graphics rendering.
"""

from .base import ISVGOptimizer
from .diffvg_optimizer import DiffVGOptimizer

__all__ = [
    "ISVGOptimizer",
    "DiffVGOptimizer"
]
