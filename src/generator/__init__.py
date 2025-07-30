# src/generator/__init__.py
"""
Generator module for text-to-image generation.

This module defines interfaces and implementations for text-to-image generation.
It uses a registry-based factory pattern to support loading different generator
models such as SDXL-Turbo or Stable Diffusion v2 via configuration.

Key components:
- `IGenerator`: Interface for all generators.
- `load_generator()`: Factory method to instantiate a generator.
- `GeneratorZoo`: Utility class to list available generators.
"""

from ..common.registry import registry
from .base import IGenerator
from .sdxl_turbo_generator import SDXLTurboGenerator
from .stable_diffusion_v2_generator import SDv2Generator

__all__ = [
    "IGenerator",
    "SDXLTurboGenerator",
    "SDv2Generator",
]


def load_generator(name, cfg: dict = None):
    """
    Factory function to instantiate a generator by name from the registry.

    Args:
        name (str): Name of the generator class to load.
        cfg (dict or None): Configuration dictionary. If provided, the generator
                            is initialized with config via `from_config()` method.

    Returns:
        IGenerator: An instance of the selected generator implementation.
    """
    generator_cls = registry.get_generator_class(name)

    # Instantiate from config if provided, else default constructor
    if cfg is not None:
        return generator_cls.from_config(cfg)
    return generator_cls()


class GeneratorZoo:
    """
    A helper class that wraps the registry mapping for available generator models.

    This provides a quick way to inspect all generator names and their variants/types
    currently registered in the system.
    """

    def __init__(self):
        # Build a simplified dictionary view of generator mappings from the registry
        self.generator_zoo = {
            k: v.__name__ for k, v in registry.mapping["generator"].items()
        }

    def __str__(self):
        """
        Pretty string representation of all available generators and their variants.
        """
        return (
            "=" * 50
            + "\n"
            + "\n".join(
                [
                    f"{name:<30} {cls}" for name, cls in self.generator_zoo.items()
                ]
            )
        )

    def __iter__(self):
        """
        Iterator over the generator zoo dictionary items.
        """
        return iter(self.generator_zoo.items())


# Singleton instance for accessing registered generators
generator_zoo = GeneratorZoo()
