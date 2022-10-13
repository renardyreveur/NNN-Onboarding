import numpy as np


def mse(out: np.array, label: np.array):
    """
    Mean Squared Error (L2 Norm)
    """
    n = out.shape[0]
    return np.sum((out - label) ** 2) / n


# TODO: Binary Cross Entropy
