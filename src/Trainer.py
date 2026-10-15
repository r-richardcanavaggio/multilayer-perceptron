from src.EarlyStopping import EarlyStopping
from src.NeuralNetwork import NeuralNetwork
from src.DataLoader import DataLoader
from src.History import History
from src.Math import binary_cross_entropy, compute_accuracy
import numpy as np


class Trainer:
    def __init__(self, model: NeuralNetwork,
                 early_stop: EarlyStopping | None, epochs: int = 80):
        self.model = model
        self.early_stop = early_stop
        self.epochs = epochs

    def fit(self, train_dataloader: DataLoader,
            val_dataloader: DataLoader) -> History:
        history = History()

        for epoch in range(self.epochs):
            train_loss: float = 0.0
            train_acc: float = 0.0

            for x_batch, y_batch in train_dataloader:
                forward_pass = self.model.forward(x_batch)

                train_loss += binary_cross_entropy(y_batch, forward_pass)
                train_acc += compute_accuracy(y_batch, forward_pass)

                d_Z = forward_pass - np.column_stack((1 - y_batch, y_batch))
                self.model.backward(d_Z)

            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)
            val_loss, val_acc = self.evaluate(val_dataloader)

            history.add_record(train_loss, val_loss, train_acc, val_acc)
            history.add_theta(self.model.get_weights_vector())
            history.set_total_epoch(epoch + 1)

            history.print_history(epoch)

            if self.early_stop and self.early_stop.should_stop(val_loss):
                break

        return history

    def evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        loss: float = 0.0
        accuracy: float = 0.0

        for x_batch, y_batch in dataloader:
            forward = self.model.forward(x_batch)

            loss += binary_cross_entropy(y_batch, forward)
            accuracy += compute_accuracy(y_batch, forward)

        return (loss / len(dataloader), accuracy / len(dataloader))
