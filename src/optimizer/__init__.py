# src/optimizer/__init__.py
"""
Optimizer module for SVG optimization using DiffVG.

This module provides SVG optimization capabilities using differentiable
vector graphics rendering.
"""

from .base import IOptimizer
from .diffvg_optimizer import DiffVGOptimizer
from .factory import OptimizerFactory

# Auto-register available optimizer
OptimizerFactory.register("diffvg", DiffVGOptimizer)

__all__ = [
    "IOptimizer",
    "DiffVGOptimizer"
    "OptimizerFactory"
]
