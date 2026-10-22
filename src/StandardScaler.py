from src.type import Tensor
import numpy as np


class StandardScaler:
    def fit(self, x: Tensor) -> None:
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std[self.std == 0] = 1.0

    def transform(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def fit_transform(self, x: Tensor) -> Tensor:
        self.fit(x)
        return self.transform(x)
