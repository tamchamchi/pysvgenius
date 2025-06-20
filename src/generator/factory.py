from typing import Optional, Type

from .base import ITextToImageGenerator


class TextToImageGeneratorFactory:
    default_strategy = "sdxl-turbo"
    strategy_map: dict[str, Type[ITextToImageGenerator]] = {}

    @classmethod
    def register(cls, name: str, generator_cls: Optional[ITextToImageGenerator]):
        if not issubclass(generator_cls, ITextToImageGenerator):
            raise TypeError(
                f"{generator_cls} must inherit from ITextToImageGenerator")
        cls.strategy_map[name] = generator_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> ITextToImageGenerator:
        name = name or cls.default_strategy
        generator_cls = cls.default_strategy.get(name)
        if generator_cls is None:
            raise TypeError(f"No generator registered with name '{name}'")
        return generator_cls(**kwargs)
