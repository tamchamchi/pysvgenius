from .base import IConverter
from typing import Type


class ConverterFactory:
    """
    A factory class for creating instances of image-to-SVG converters
    based on registered strategy names.

    This follows the Factory + Registry design pattern, allowing you to
    dynamically register and instantiate different implementations of
    IConverter (e.g., Vtracer, Potrace, etc.).

    Attributes:
        default_strategy (str): The default converter strategy name used if none is specified.
        strategy_map (dict[str, Type[IConverter]]): Mapping from strategy names to converter classes.
    """

    default_strategy: str = "vtracer"
    strategy_map: dict[str, Type[IConverter]] = {}

    @classmethod
    def register(cls, name: str, converter_cls: Type[IConverter]):
        """
        Register a new converter class under a given strategy name.

        Args:
            name (str): The strategy name to register (e.g., "vtracer").
            converter_cls (Type[IConverter]): A class implementing the IConverter interface.

        Raises:
            TypeError: If the provided class does not inherit from IConverter.
        """
        if not issubclass(converter_cls, IConverter):
            raise TypeError(
                f'{converter_cls} must inherit from IConverter')
        cls.strategy_map[name] = converter_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> IConverter:
        """
        Create an instance of a registered image-to-converter strategy.

        Args:
            name (str): The name of the registered converter strategy to use.
                        If None or empty, the default strategy will be used.
            **kwargs: Additional keyword arguments passed to the converter's constructor.

        Returns:
            IConverter: An instance of the requested converter class.

        Raises:
            TypeError: If no converter is registered under the provided name.
        """
        name = name or cls.default_strategy
        converter_cls = cls.strategy_map.get(name)
        if converter_cls is None:
            raise TypeError(f"No converter registered with name '{name}'")
        return converter_cls(**kwargs)
