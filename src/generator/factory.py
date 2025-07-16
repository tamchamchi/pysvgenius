from typing import Optional, Type

from .base import ITextToImageGenerator


class TextToImageGeneratorFactory:
    """
    A factory class for creating instances of text-to-image generators
    based on registered strategy names.

    This follows the Factory + Registry design pattern, allowing you to
    dynamically register and instantiate different implementations of
    `ITextToImageGenerator`.

    Attributes:
        default_strategy (str): The default generator strategy name.
        strategy_map (dict[str, Type[ITextToImageGenerator]]): Mapping of strategy names to generator classes.
    """

    default_strategy = "sdxl-turbo"
    strategy_map: dict[str, Type[ITextToImageGenerator]] = {}

    @classmethod
    def register(cls, name: str, generator_cls: Optional[ITextToImageGenerator]):
        """
        Register a new generator class under a given strategy name.

        Args:
            name (str): The strategy name to associate with the generator class.
            generator_cls (Type[ITextToImageGenerator]): A class that implements the ITextToImageGenerator interface.

        Raises:
            TypeError: If the provided class does not inherit from ITextToImageGenerator.
        """
        if not issubclass(generator_cls, ITextToImageGenerator):
            raise TypeError(
                f"{generator_cls} must inherit from ITextToImageGenerator")
        cls.strategy_map[name] = generator_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> ITextToImageGenerator:
        """
        Create an instance of a registered text-to-image generator strategy.

        Args:
            name (str): The name of the generator strategy to instantiate.
            **kwargs: Optional keyword arguments passed to the generator's constructor.

        Returns:
            ITextToImageGenerator: An instance of the requested generator class.

        Raises:
            TypeError: If no generator is registered under the provided name.
        """
        name = name or cls.default_strategy
        generator_cls = cls.strategy_map.get(name)
        if generator_cls is None:
            raise TypeError(f"No generator registered with name '{name}'")
        return generator_cls(**kwargs)
