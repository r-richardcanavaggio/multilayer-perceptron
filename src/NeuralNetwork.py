import numpy as np
from typing import TypedDict
from numpy.typing import NDArray
from src.DenseLayer import DenseLayer
from src.Math import binary_cross_entropy, compute_accuracy
from src.EarlyStopping import EarlyStopping


class HistoryDict(TypedDict):
    loss: list[float]
    val_loss: list[float]
    acc: list[float]
    val_acc: list[float]
    thetas: list[NDArray[np.float64]]
    total_epoch: int


class NeuralNetwork:
    def __init__(self, input_size: int,
                 layers_sizes: list[int], learning_rate: float,
                 activation: str, optimizer: str):
        self.learning_rate = learning_rate
        self.layers: list[DenseLayer] = []

        if not layers_sizes:
            return

        for layer in layers_sizes:
            new_layer = DenseLayer(
                input_size, layer, activation=activation,
                weights_initializer='glorot', optimizer=optimizer
            )
            self.layers.append(new_layer)
            input_size = layer
        self.layers.append(DenseLayer(
            layers_sizes[-1], 2, activation='softmax',
            weights_initializer='glorot', optimizer=optimizer
        ))

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.feed_forward(out)
        return out

    def backward(self, d_Z: np.ndarray) -> np.ndarray:
        out = d_Z
        for layer in reversed(self.layers):
            out = layer.backward(out, self.learning_rate)
        return out

    def export(self, path: str):
        nn_dict: dict[str, NDArray[np.float64] | str] = {}

        for i, layer in enumerate(self.layers):
            nn_dict[f"W_{i}"] = layer.weights
            nn_dict[f"b_{i}"] = layer.biases
            nn_dict[f"A_{i}"] = layer.activation

        try:
            np.savez(path, allow_pickle=True, **nn_dict)
        except PermissionError as error:
            print(error)
        except OSError as error:
            print(error)
        except ValueError as error:
            print(error)
        else:
            print(f"Model saved succesfully in {path}")

    def fit(self, input_matrix: np.ndarray, y: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int, batch_size: int,
            early_stop: EarlyStopping | None) -> HistoryDict:

        history: HistoryDict = {
            'loss': [],
            'val_loss': [],
            'acc': [],
            'val_acc': [],
            'thetas': [],
            'total_epoch': 0
            }

        width = len(str(epochs))

        # Output in a binary format, M = [0, 1] B = [1, 0]
        y_double = np.column_stack((1 - y, y))

        for epoch in range(epochs):
            total_loss_train: float = 0.0
            total_accuracy_train: float = 0.0
            num_batches = 0

            shuffled_indices = np.random.permutation(input_matrix.shape[0])
            input_matrix = input_matrix[shuffled_indices]
            y = y[shuffled_indices]
            y_double = y_double[shuffled_indices]

            for first in range(0, len(input_matrix), batch_size):
                input_matrix_batch = input_matrix[first: first + batch_size]
                y_batch = y[first: first + batch_size]
                y_double_batch = y_double[first: first + batch_size]

                forward_pass = self.forward(input_matrix_batch)

                loss = binary_cross_entropy(y_batch, forward_pass)
                total_loss_train += loss

                accuracy = compute_accuracy(y_batch, forward_pass)
                total_accuracy_train += accuracy

                num_batches += 1

                d_Z = forward_pass - y_double_batch
                self.backward(d_Z)

            total_loss_train /= num_batches
            total_accuracy_train /= num_batches

            history['loss'].append(total_loss_train)

            history['acc'].append(total_accuracy_train)

            pred_validation = self.forward(X_val)

            loss_val = binary_cross_entropy(y_val, pred_validation)
            history['val_loss'].append(loss_val)

            accuracy_val = compute_accuracy(y_val, pred_validation)
            history['val_acc'].append(accuracy_val)

            history['thetas'].append(self.get_weights_vector())

            print(f"Epoch {epoch + 1:>{width}}: "
                  f"Train loss = {total_loss_train:.8f} | "
                  f"Validation loss = {loss_val:.8f}")

            history['total_epoch'] = epoch + 1

            if early_stop and early_stop.should_stop(loss_val):
                break

        return history

    @classmethod
    def from_npz(cls, filename: str):
        loaded_layers = np.load(filename)
        layers: list[DenseLayer] = []

        for i in range(len(loaded_layers) // 3):
            weights = loaded_layers[f"W_{i}"]
            biases = loaded_layers[f"b_{i}"]
            activation = loaded_layers[f"A_{i}"]
            layer = DenseLayer.from_computed(weights, biases, activation)
            layers.append(layer)
        instance = cls(0, [], 0, "", "")
        instance.layers = layers
        return instance

    def get_weights_vector(self) -> np.ndarray:
        return np.concatenate(
            [p.flatten() for layer in self.layers for p in (layer.weights, layer.biases)]
            )

    def set_weights_vector(self, theta: np.ndarray):
        index = 0

        for layer in self.layers:
            size = layer.weights.size
            chunk = theta[index: index + size]
            chunk = np.reshape(chunk, layer.weights.shape)
            layer.weights = chunk
            index += size

            size = layer.biases.size
            chunk = theta[index: index + size]
            chunk = np.reshape(chunk, layer.biases.shape)
            layer.biases = chunk
            index += size
