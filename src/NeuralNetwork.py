import numpy as np
from numpy.typing import NDArray
from src.DenseLayer import DenseLayer
from src.Math import binary_cross_entropy, compute_accuracy


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
            np.savez(path, kwargs='nn_dict')
        except PermissionError as error:
            print(error)
        except OSError as error:
            print(error)
        except ValueError as error:
            print(error)
        else:
            print(f"Model saved succesfully in {path}")

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int, batch_size: int,
            stop_early: int
            ) -> dict[str, list[float | np.ndarray] | int]:

        history: dict[str, list[float | np.ndarray] | int] = {
            'loss': [],
            'val_loss': [],
            'acc': [],
            'val_acc': [],
            'thetas': []
            }

        width = len(str(epochs))
        patience = stop_early
        best_val_loss = float('inf')
        count_loss = 0

        # Output in a binary format, M = [0, 1] B = [1, 0]
        y_double = np.column_stack((1 - y, y))

        for epoch in range(epochs):
            total_loss_train = 0
            total_accuracy_train = 0.
            num_batches = 0


            for first in range(0, len(X), batch_size):
                X_batch = X[first: first + batch_size]
                y_batch = y[first: first + batch_size]
                y_double_batch = y_double[first: first + batch_size]

                forward_pass = self.forward(X_batch)

                loss = binary_cross_entropy(y_batch, forward_pass)
                total_loss_train += loss

                accuracy = compute_accuracy(y_batch, forward_pass)
                total_accuracy_train += accuracy

                num_batches += 1

                d_Z = forward_pass - y_double_batch
                self.backward(d_Z)

            total_loss_train /= num_batches
            total_accuracy_train /= num_batches

            if isinstance(history['loss'], list):
                history['loss'].append(total_loss_train)

            if isinstance(history['acc'], list):
                history['acc'].append(total_accuracy_train)

            pred_validation = self.forward(X_val)

            loss_val = binary_cross_entropy(y_val, pred_validation)
            if isinstance(history['val_loss'], list):
                history['val_loss'].append(loss_val)

            accuracy_val = compute_accuracy(y_val, pred_validation)
            if isinstance(history['val_acc'], list):
                history['val_acc'].append(accuracy_val)

            if isinstance(history['thetas'], list):
                history['thetas'].append(self.get_weights_vector())

            print(f"Epoch {epoch + 1:>{width}}: "
                  f"Train loss = {total_loss_train:.8f} | "
                  f"Validation loss = {loss_val:.8f}")

            history['total_epoch'] = epoch + 1

            if stop_early:
                if loss_val < best_val_loss:
                    count_loss = 0
                    best_val_loss = loss_val
                else:
                    count_loss += 1

                if count_loss == patience:
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
            morceau = theta[index: index + size]
            morceau = np.reshape(morceau, layer.weights.shape)
            layer.weights = morceau
            index += size

            size = layer.biases.size
            morceau = theta[index: index + size]
            morceau = np.reshape(morceau, layer.biases.shape)
            layer.biases = morceau
            index += size
