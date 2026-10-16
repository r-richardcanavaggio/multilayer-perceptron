from collections.abc import Iterator
from src.Dataset import Dataset
from src.type import Tensor
import numpy as np


class DataLoader:
    def __init__(self, input_set: Dataset,
                 batch_size: int = 8, shuffle: bool = True):
        self.input_set = input_set
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return int(np.ceil(len(self.input_set) / self.batch_size))

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        n = len(self.input_set)

        indices = np.random.permutation(n) if self.shuffle else np.arange(n)

        for first in range(0, n, self.batch_size):
            batch_idx = indices[first: first + self.batch_size]
            parameters_batch, labels_batch = self.input_set[batch_idx]
            yield parameters_batch, labels_batch
