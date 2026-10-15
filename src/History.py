from typing import TypedDict
from numpy.typing import NDArray
from numpy import float64


class HistoryDict(TypedDict):
    loss: list[float]
    val_loss: list[float]
    accuracy: list[float]
    val_accuracy: list[float]
    thetas: list[NDArray[float64]]
    total_epoch: int


class History:
    def __init__(self):
        self.history: HistoryDict = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'thetas': [],
            'total_epoch': 0
        }

    def add_theta(self, theta: NDArray[float64]):
        self.history['thetas'].append(theta)

    def set_total_epoch(self, total_epoch: int):
        self.history['total_epoch'] = total_epoch

    def add_record(self, loss: float, val_loss: float,
                   acc: float, val_acc: float):
        self.history['loss'].append(loss)
        self.history['val_loss'].append(val_loss)
        self.history['accuracy'].append(acc)
        self.history['val_accuracy'].append(val_acc)

    def print_history(self, epoch: int):
        print(f"Epoch {epoch + 1}: "
              f"Training loss = {self.history['loss'][epoch]:.8f} | "
              f"Validation loss = {self.history['val_loss'][epoch]:.8f}")
