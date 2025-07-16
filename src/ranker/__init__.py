# src/ranker/__init__.py
"""
Ranker module for SVG ranking and selection.

This module provides SVG ranking capabilities using various strategies
like aesthetic ranking and text-image similarity ranking. It includes
a factory pattern for easy instantiation of different ranker implementations.
"""

from .base import ISVGRanker
from .factory import SVGRankerFactory
from .aesthetic_ranker import AestheticRanker
from .siglip_ranker import SigLipRanker
# from .paligemma_ranker import 

# Auto-register available rankers
SVGRankerFactory.register("aesthetic", AestheticRanker)
SVGRankerFactory.register("siglip", SigLipRanker)
# SVGRankerFactory.register("paligemma", PaligemmaRanker)

__all__ = [
    "ISVGRanker",
    "SVGRankerFactory",
    "AestheticRanker",
    "SigLipRanker",
    "PaligemmaRanker"
]
