from abc import ABC, abstractmethod
from PIL import Image


class IOptimizer(ABC):
    @abstractmethod
    def process(svg: str, image: Image, args: dict, limit: int, **kwargs):
        pass

    def __call__(self, svg: str, image: Image, args: dict, limit: int):
        return self.process(svg, image, args, limit)
