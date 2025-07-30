# src/converter/__init__.py
"""
Converter module for image-to-SVG conversion.

This module provides image-to-SVG conversion capabilities using various algorithms
like VTracer. It includes a factory pattern for easy instantiation of different
converter implementations.
"""

# Import interface and concrete converter classes
# Interface base class for all converters
from .base import IConverter
from .vtracer_binary_search import VtracerBinarySearch
from .vtracer_grid_search import VtracerGribSearch

# Import the registry for managing converter registration and lookup
from ..common import registry

# Define public API of the module
__all__ = [
    "IConverter",
    "VtracerBinarySearch",
    "VtracerGribSearch"
]


def load_converter(name):
    """
    Load a converter instance by name using the registry.

    Args:
        name (str): The name of the registered converter.

    Returns:
        An instance of the corresponding converter class.
    """
    converter_cls = registry.get_converter_class(name)
    return converter_cls()  # Instantiate and return


class ConverterZoo:
    """
    A utility class to list all available converter classes registered in the registry.
    Acts like a model zoo but for converter implementations.
    """

    def __init__(self):
        # Build a dictionary with {registered_name: class_name}
        self.converter_zoo = {
            k: v.__name__ for k, v in registry.mapping["converter"].items()
        }

    def __str__(self):
        """
        Pretty string representation of all available converters and their variants.

        Returns:
            str: A formatted list of available converters.
        """
        return (
            "=" * 50
            + "\n"
            + "\n".join(
                [
                    f"{name:<30} {cls}" for name, cls in self.converter_zoo.items()
                ]
            )
        )

    def __iter__(self):
        """
        Allow iteration over the registered converters.

        Yields:
            tuple: (name, class_name)
        """
        return iter(self.converter_zoo.items())


# Instantiate a global zoo object to access available converters
converter_zoo = ConverterZoo()
