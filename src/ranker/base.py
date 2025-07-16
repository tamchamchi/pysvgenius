from abc import ABC, abstractmethod


class ISVGRanker(ABC):
    @abstractmethod
    def process(self, svgs: list[str], top_k: int, **kwargs):
        pass

    def __call__(self, svgs: list[str], top_k: int, **kwargs):
        self.process(svgs, top_k, **kwargs)
