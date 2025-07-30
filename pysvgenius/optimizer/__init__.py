# src/optimizer/__init__.py
"""
Optimizer module for SVG optimization using DiffVG.

This module provides SVG optimization capabilities using differentiable
vector graphics rendering.
"""
import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path

from ..common import registry
from ..utils.logger import create_console_logger
from .base import IOptimizer
from .components.aesthetic_evaluator_torch import AestheticEvaluatorTorch
from .components.image_processor_torch import ImageProcessorTorch
from .diffvg_optimizer import DiffVGOptimizer

__all__ = [
    "IOptimizer",
    "DiffVGOptimizer",
    "AestheticEvaluatorTorch",
    "ImageProcessorTorch"
]
# Check if pydiffvg is installed
if importlib.util.find_spec("pydiffvg") is None:
    print("⚠ pydiffvg is not installed. Attempting to build from source...")
    base_dir = os.path.dirname(__file__)
    diffvg_dir = os.path.join(base_dir, "diffvg")

    # Clone the diffvg repository if not present
    if not os.path.exists(diffvg_dir):
        print("Cloning diffvg repository...")
        subprocess.check_call(["git", "clone", "https://github.com/BachiLi/diffvg.git", diffvg_dir])
        print("Initializing and updating git submodules...")
        subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=diffvg_dir)
    else:
        print(f"Found existing diffvg at {diffvg_dir}, skipping clone.")

    # Build and install pydiffvg into the current environment
    print("Building and installing pydiffvg...")
    subprocess.check_call([sys.executable, "setup.py", "install"], cwd=diffvg_dir)

    print("✅ pydiffvg installation attempt completed. Try importing again.")

def load_optimizer(name, cfg: dict = {}):
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
    if cfg.get("aesthetic_model_path") is None or cfg.get("clip_model_path") is None:
        model_path = Path(registry.get_path("model_dir") +
                          "/sac+logos+ava1-l14-linearMSE.pth")
        clip_model_path = Path(registry.get_path(
            "model_dir") + "/clip/ViT-L-14.pt")
    else:
        model_path = Path(cfg.get("aesthetic_model_path"))
        clip_model_path = Path(cfg.get("clip_model_path"))

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
