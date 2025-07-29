# src/optimizer/__init__.py
"""
Optimizer module for SVG optimization using DiffVG.

This module provides SVG optimization capabilities using differentiable
vector graphics rendering.
"""

from .base import IOptimizer
from .diffvg_optimizer import DiffVGOptimizer
from .components.aesthetic_evaluator_torch import AestheticEvaluatorTorch
from .components.image_processor_torch import ImageProcessorTorch
from src.common.registry import registry
from src.utils.logger import create_console_logger
import logging
from pathlib import Path

__all__ = [
    "IOptimizer",
    "DiffVGOptimizer",
    "AestheticEvaluatorTorch",
    "ImageProcessorTorch"
]


def load_optimizer(name):
    """
    Factory function to instantiate a optimizer by name from the registry.

    Args:
        name (str): Name of the optimizer class to load.
        cfg (dict or None): Configuration dictionary. If provided, the optimizer
                            is initialized with config via `from_config()` method.

    Returns:
        Ioptimizer: An instance of the selected optimizer implementation.
    """
    optimizer_cls = registry.get_optimizer_class(name)

    model_path = Path(registry.get_path("model_dir") + "/sac+logos+ava1-l14-linearMSE.pth")
    clip_model_path = Path(registry.get_path("model_dir") + "/clip/ViT-L-14.pt")

    aesthetic_eval_torch = AestheticEvaluatorTorch(model_path, clip_model_path)
    image_processor_torch = ImageProcessorTorch()
    image_processor_torch_ref = ImageProcessorTorch()

    logger = create_console_logger("optimize", logging.INFO)
    optimizer = optimizer_cls(
        aesthetic_evaluator_torch=aesthetic_eval_torch,
        siglip_model=None,
        image_processor_torch=image_processor_torch,
        image_processor_torch_ref=image_processor_torch_ref,
        logger=logger
    )

    return optimizer
