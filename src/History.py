from typing import TypedDict
from src.type import Tensor
import matplotlib.pyplot as plt


class HistoryDict(TypedDict):
    loss: list[float]
    val_loss: list[float]
    accuracy: list[float]
    val_accuracy: list[float]
    thetas: list[Tensor]
    total_epoch: int


class History:
    def __init__(self) -> None:
        self.history: HistoryDict = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'thetas': [],
            'total_epoch': 0
        }

    def add_theta(self, theta: Tensor) -> None:
        self.history['thetas'].append(theta)

    def set_total_epoch(self, total_epoch: int) -> None:
        self.history['total_epoch'] = total_epoch

    def add_record(self, loss: float, val_loss: float,
                   acc: float, val_acc: float) -> None:
        self.history['loss'].append(loss)
        self.history['val_loss'].append(val_loss)
        self.history['accuracy'].append(acc)
        self.history['val_accuracy'].append(val_acc)

    def print_history(self, epoch: int) -> None:
        print(f"Epoch {epoch + 1}: "
              f"Training loss = {self.history['loss'][epoch]:.8f} | "
              f"Validation loss = {self.history['val_loss'][epoch]:.8f}")

    def plot_loss(self) -> None:
        x = [x for x in range(self.history['total_epoch'])]
        plt.plot(x, self.history['loss'], color='blue', label='training loss')
        plt.plot(x, self.history['val_loss'],
                 color='orange', linestyle='--',
                 label='validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('loss.png')

    def plot_acc(self) -> None:
        x = [i for i in range(self.history['total_epoch'])]
        plt.plot(x, self.history['accuracy'],
                 color='blue', label='training accuracy')
        plt.plot(x, self.history['val_accuracy'],
                 color='orange', linestyle='--',
                 label='validation accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig('accuracy.png')
