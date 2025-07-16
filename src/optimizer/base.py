from abc import ABC, abstractmethod

class IOptimizer(ABC):
    @abstractmethod
    def process(svg_code: str):
        pass