from abc import ABC, abstractmethod

class ISVGOptimizer(ABC):
    @abstractmethod
    def process(svg_code: str):
        pass