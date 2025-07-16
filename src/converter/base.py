from abc import ABC, abstractmethod
from PIL import Image


class IConverter(ABC):
    @abstractmethod
    def process(self, images: list[Image.Image]):
        pass

    def __call__(self, images: list[Image.Image], **kwargs):
        return self.process(images, **kwargs)
