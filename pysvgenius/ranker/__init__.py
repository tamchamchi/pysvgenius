# src/ranker/__init__.py
"""
Ranker module for SVG ranking and selection.

This module provides SVG ranking capabilities using various strategies
like aesthetic ranking and text-image similarity ranking. It includes
a factory pattern for easy instantiation of different ranker implementations.
"""

from .base import IRanker
from .aesthetic_ranker import AestheticRanker
from .siglip_ranker import SigLipRanker
from ..common import registry

__all__ = [
    "IRanker",
    "AestheticRanker",
    "SigLipRanker",
    "PaligemmaRanker"
]


def load_ranker(name, cfg: dict = None):
    """
    Factory function to instantiate a ranker by name from the registry.

    Args:
        name (str): Name of the ranker class to load.
        cfg (dict or None): Configuration dictionary. If provided, the ranker
                            is initialized with config via `from_config()` method.

    Returns:
        Iranker: An instance of the selected ranker implementation.
    """
    ranker_cls = registry.get_ranker_class(name)

    # Instantiate from config if provided, else default constructor
    if cfg is not None:
        return ranker_cls.from_config(cfg)
    return ranker_cls()
