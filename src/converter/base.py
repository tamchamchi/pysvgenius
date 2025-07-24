from abc import ABC, abstractmethod
from PIL import Image


class IConverter(ABC):
    @abstractmethod
    def process(self, images: list[Image.Image], limit: int):
        pass

    def __call__(self, images: list[Image.Image], limit: int = 10000, **kwargs):
        return self.process(images, limit, **kwargs)
