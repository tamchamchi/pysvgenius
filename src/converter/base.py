from abc import ABC, abstractmethod
from PIL import Image


class IImageToConverter(ABC):
    @abstractmethod
    def process(self, images: list[Image.Image]):
        pass

    def __call__(self, images: list[Image.Image], **kwargs):
        self.process(images, **kwargs)
