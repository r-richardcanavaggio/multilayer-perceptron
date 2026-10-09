import numpy as np
from src.DenseLayer import DenseLayer
from src.Math import binary_cross_entropy, compute_accuracy


class NeuralNetwork:
    def __init__(self, input_size: int,
                 layers_sizes: list[int], learning_rate: float,
                 activation: str, optimizer: str):
        self.learning_rate = learning_rate
        self.layers = []

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

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.feed_forward(out)
        return out

    def backward(self, d_Z):
        out = d_Z
        for layer in reversed(self.layers):
            out = layer.backward(out, self.learning_rate)
        return out

    def export(self, path: str):
        nn_dict = {}
        for i, layer in enumerate(self.layers):
            nn_dict[f"W_{i}"] = layer.weights
            nn_dict[f"b_{i}"] = layer.biases
            nn_dict[f"A_{i}"] = layer.activation

        try:
            np.savez(path, **nn_dict)
        except PermissionError as error:
            print(error)
        except OSError as error:
            print(error)
        except ValueError as error:
            print(error)
        else:
            print(f"Model saved succesfully in {path}")

    def fit(self,
            X: np.ndarray, y: np.ndarray,
            y_double: np.ndarray, X_val: np.ndarray,
            y_val: np.ndarray,
            epochs: int, batch_size: int
            ) -> dict:
        history = {'loss': [],
                   'val_loss': [],
                   'acc': [],
                   'val_acc': []}

        width = len(str(epochs))

        PATIENCE = 10
        best_val_loss = 0
        count_loss = 0

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
            history['loss'].append(total_loss_train)
            history['acc'].append(total_accuracy_train)

            pred_validation = self.forward(X_val)

            loss_val = binary_cross_entropy(y_val, pred_validation)
            history['val_loss'].append(loss_val)

            accuracy_val = compute_accuracy(y_val, pred_validation)
            history['val_acc'].append(accuracy_val)

            if epoch == 1:
                best_val_loss = loss_val

            if loss_val < best_val_loss:
                count_loss = 0
                best_val_loss = loss_val
            else:
                count_loss += 1

            if count_loss == PATIENCE:
                return history

            print(f"Epoch {epoch + 1:>{width}}: "
                  f"Train loss = {total_loss_train:.8f} | "
                  f"Validation loss = {loss_val:.8f}")
        return history

    @classmethod
    def from_npz(cls, filename: str):
        loaded_layers = np.load(filename)

        layers = []
        for i in range(len(loaded_layers) // 3):
            weights = loaded_layers[f"W_{i}"]
            biases = loaded_layers[f"b_{i}"]
            activation = loaded_layers[f"A_{i}"]
            layer = DenseLayer.from_computed(weights, biases, activation)
            layers.append(layer)
        return cls(layers, None)
