from src.ft_load import ft_load
from src.type import Tensor
from src.StandardScaler import StandardScaler
from typing import Any
from numpy import float64


class Dataset:
    def __init__(self, path: str):
        data = ft_load(path)

        raw_labels = data.iloc[:, 1].replace({'M': 1, 'B': 0})

        self.labels = raw_labels.to_numpy(dtype=float64)
        self.parameters = data.iloc[:, 2:].to_numpy(dtype=float64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: Any) -> tuple[Tensor, Tensor]:
        return self.parameters[idx], self.labels[idx]

    @property
    def columns(self) -> int:
        return self.parameters.shape[1]

    def normalize(self, scaler: StandardScaler) -> None:
        self.parameters = scaler.transform(self.parameters)
