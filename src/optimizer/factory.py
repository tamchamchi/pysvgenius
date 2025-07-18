from .base import IOptimizer
from typing import Type


class OptimizerFactory:
    """
    Factory class for registering and creating SVG optimizer strategies.

    This class implements the Factory design pattern, allowing you to register
    multiple SVG optimizer implementations (subclasses of `IOptimizer`)
    and create instances of them dynamically by name.

    Attributes
    ----------
    default_strategy : str
        The default optimizer strategy name to use when none is provided.
    strategy_map : dict[str, Type[IOptimizer]]
        Mapping of strategy names to optimizer classes.
    """

    default_strategy: str = "diffvg"
    strategy_map: dict[str, Type[IOptimizer]] = {}

    @classmethod
    def register(cls, name: str, optimizer_cls: Type[IOptimizer]):
        """
        Register a new SVG optimizer class under a strategy name.

        Parameters
        ----------
        name : str
            The name used to identify the optimizer strategy.
        optimizer_cls : Type[IOptimizer]
            The optimizer class to register. Must be a subclass of `IOptimizer`.

        Raises
        ------
        TypeError
            If the provided class is not a subclass of `IOptimizer`.
        """
        if not issubclass(optimizer_cls, IOptimizer):
            raise TypeError(
                f'{optimizer_cls} must inherit from IOptimizer')
        cls.strategy_map[name] = optimizer_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> IOptimizer:
        """
        Create an instance of a registered optimizer strategy.

        Parameters
        ----------
        name : str
            The name of the optimizer strategy to use. If None or empty,
            the default strategy will be used.
        **kwargs : dict
            Keyword arguments passed to the optimizer's constructor.

        Returns
        -------
        IOptimizer
            An instance of the requested optimizer strategy.

        Raises
        ------
        TypeError
            If no optimizer is registered under the given name.
        """
        name = name or cls.default_strategy
        optimizer_cls = cls.strategy_map.get(name)
        if optimizer_cls is None:
            raise TypeError(f"No optimizer registered with name '{name}'")
        return optimizer_cls(**kwargs)
