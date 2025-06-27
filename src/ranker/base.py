from abc import ABC, abstractmethod


class ISVGRanker(ABC):
    @abstractmethod
    def process(svg_list: list[str], prompt: str = None):
        pass
