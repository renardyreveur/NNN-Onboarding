from abc import ABC, abstractmethod


class Module(ABC):
    """
    Abstract Class for defining Neural Network Layers
    Makes the subclass a callable (for ease of 'forward pass' notation)
    """
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward pass should be implemented!")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
