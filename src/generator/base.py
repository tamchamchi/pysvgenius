from abc import ABC, abstractmethod

class ITextToImageGenerator(ABC):
    @abstractmethod
    def process(prompt: str, **kwargs):
        pass