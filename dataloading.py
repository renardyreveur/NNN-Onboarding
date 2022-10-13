import random

import numpy as np


class XORDataset:
    def __init__(self, batch_size, num_batches_per_epoch):
        self.data = [
            ([[0], [0]], [[0]]),
            ([[1], [0]], [[1]]),
            ([[0], [1]], [[1]]),
            ([[1], [1]], [[0]]),
        ]
        random.shuffle(self.data)
        self.data, self.label = zip(*self.data)
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

    def __iter__(self):
        for _ in range(self.num_batches_per_epoch):
            choices = np.random.choice(np.arange(0, len(self.data)), self.batch_size)
            yield np.stack(np.asarray(self.data)[choices], axis=0), np.stack(np.asarray(self.label)[choices], axis=0)
