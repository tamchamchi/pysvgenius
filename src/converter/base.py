from abc import ABC, abstractmethod
from PIL import Image
class IImageToConverter(ABC):
    @abstractmethod
    def process(self, images: list[Image.Image]):
        pass