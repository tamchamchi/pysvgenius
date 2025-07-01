from .base import ISVGRanker
from typing import Type


class SVGRankerFactory:
    """
    A factory class for creating instances of SVG ranking strategies
    that implement the ISVGRanker interface.

    This follows the Factory + Registry pattern, allowing flexible
    selection and registration of different ranking algorithms.

    Attributes:
        default_strategy (str): Default ranker name to use when none is specified.
        strategy_map (dict[str, Type[ISVGRanker]]): Mapping of ranker names to their implementation classes.
    """

    default_strategy: str = "multi"
    strategy_map: dict[str, Type[ISVGRanker]] = {}

    @classmethod
    def register(cls, name: str, ranker_cls: Type[ISVGRanker]):
        """
        Register a new SVG ranker class under a specific name.

        Args:
            name (str): The strategy name to associate with the ranker.
            ranker_cls (Type[ISVGRanker]): A class implementing the ISVGRanker interface.

        Raises:
            TypeError: If ranker_cls does not inherit from ISVGRanker.
        """
        if not issubclass(ranker_cls, ISVGRanker):
            raise TypeError(
                f'{ranker_cls} must inherit from ISVGRanker')
        cls.strategy_map[name] = ranker_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> ISVGRanker:
        """
        Create an instance of a registered SVG ranker strategy.

        Args:
            name (str): The name of the registered ranker strategy to use.
                        If None or empty, the default strategy is used.
            **kwargs: Additional arguments to pass to the ranker constructor.

        Returns:
            ISVGRanker: An instance of the selected ranker.

        Raises:
            TypeError: If no ranker is registered under the provided name.
        """
        name = name or cls.default_strategy
        ranker_cls = cls.strategy_map.get(name)
        if ranker_cls is None:
            raise TypeError(f"No ranker registered with name '{name}'")
        return ranker_cls(**kwargs)
