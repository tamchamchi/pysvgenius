# pysvgenius/__init__.py
"""
PysvgGenius - A comprehensive library for text-to-SVG generation and optimization.

This library provides a complete pipeline for generating and optimizing SVG graphics
from text prompts using state-of-the-art AI models and vector graphics optimization.

Main components:
- Generator: Text-to-image generation using models like SDXL-Turbo
- Converter: Image-to-SVG conversion using algorithms like VTracer  
- Ranker: SVG ranking using aesthetic or text-similarity models
- Optimizer: SVG optimization using differentiable vector graphics
- Utils: Common utilities for image processing, logging, etc.
"""

# Import main components from src
from src.generator import GeneratorFactory, SDXLTurboGenerator
from src.converter import ConverterFactory, VtracerConverter  
from src.ranker import RankerFactory, AestheticRanker, SigLipRanker
from src.optimizer import DiffVGOptimizer
from src.utils import get_library_logger, create_console_logger, svg_to_png

__version__ = "0.1.0"
__author__ = "PysvgGenius Team"
__email__ = "contact@pysvgenius.com"

__all__ = [
    # Factories
    "GeneratorFactory",
    "ConverterFactory", 
    "RankerFactory",
    
    # Implementations
    "SDXLTurboGenerator",
    "VtracerConverter",
    "AestheticRanker", 
    "SigLipRanker",
    "DiffVGOptimizer",
    
    # Utilities
    "get_library_logger",
    "create_console_logger",
    "svg_to_png",
    
    # Metadata
    "__version__",
    "__author__", 
    "__email__",
]
