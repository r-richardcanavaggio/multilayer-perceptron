from src.ft_load import ft_load
from src.type import Tensor
from typing import Any


class Dataset:
    def __init__(self, path: str):
        data = ft_load(path)

        self.labels = data.iloc[:, 0].to_numpy()
        self.parameters = data.iloc[:, 1:].to_numpy()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: Any) -> tuple[Tensor, Tensor]:
        return self.parameters[idx], self.labels[idx]

    @property
    def columns(self) -> int:
        return self.parameters.shape[1]
