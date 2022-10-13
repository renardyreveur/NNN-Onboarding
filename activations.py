import numpy as np

from base_module import Module


class ReLU(Module):
    def forward(self, x):
        return np.maximum(0, x)


class Sigmoid(Module):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
