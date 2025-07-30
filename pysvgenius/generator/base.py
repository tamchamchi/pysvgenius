from abc import ABC, abstractmethod


class IGenerator(ABC):
    @abstractmethod
    def process(prompt: str, num_images: int = 3, **kwargs):
        pass

    def __call__(self, prompt: str, num_images: int = 3, **kwargs):
        return self.process(prompt, num_images, **kwargs)
