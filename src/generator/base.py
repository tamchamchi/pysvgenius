from abc import ABC, abstractmethod


class ITextToImageGenerator(ABC):
    @abstractmethod
    def process(prompt: str, **kwargs):
        pass

    def __call__(self, prompt: str, **kwargs):
        return self.process(prompt, **kwargs)
