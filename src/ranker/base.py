from abc import ABC, abstractmethod

class ISVGRanker(ABC):
    @abstractmethod
    def process(self):
        pass

    def __call__(self, **kwargs):
        self.process(**kwargs)
