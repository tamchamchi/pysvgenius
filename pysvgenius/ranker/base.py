from abc import ABC, abstractmethod


class IRanker(ABC):
    @abstractmethod
    def process(self, svgs: list[str], prompt: str, batch_size: int, top_k: int, **kwargs):
        pass

    def __call__(self, svgs: list[str], prompt: str = None, batch_size: int = 16, top_k: int = 2, **kwargs):
        return self.process(svgs, prompt, batch_size, top_k, **kwargs)
