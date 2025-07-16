# src/ranker/__init__.py
"""
Ranker module for SVG ranking and selection.

This module provides SVG ranking capabilities using various strategies
like aesthetic ranking and text-image similarity ranking. It includes
a factory pattern for easy instantiation of different ranker implementations.
"""

from .base import IRanker
from .factory import RankerFactory
from .aesthetic_ranker import AestheticRanker
from .siglip_ranker import SigLipRanker
# from .paligemma_ranker import

# Auto-register available rankers
RankerFactory.register("aesthetic", AestheticRanker)
RankerFactory.register("siglip", SigLipRanker)
# RankerFactory.register("paligemma", PaligemmaRanker)

__all__ = [
    "IRanker",
    "RankerFactory",
    "AestheticRanker",
    "SigLipRanker",
    "PaligemmaRanker"
]
