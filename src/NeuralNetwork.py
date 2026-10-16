import numpy as np
from numpy.typing import NDArray
from src.DenseLayer import DenseLayer
from src.Optimizer import OptimizerFactory


class NeuralNetwork:
    def __init__(self, input_size: int,
                 layers_sizes: list[int], learning_rate: float,
                 activation: str, optimizer: str, **optimizer_kwargs: float):
        self.layers: list[DenseLayer] = []

        if not layers_sizes:
            return

        for layer in layers_sizes:
            opt = OptimizerFactory.create(
                optimizer, learning_rate=learning_rate, **optimizer_kwargs)
            new_layer = DenseLayer(
                input_size, layer, activation=activation,
                weights_initializer='glorot', optimizer=opt
            )
            self.layers.append(new_layer)
            input_size = layer

        opt = OptimizerFactory.create(
            optimizer, learning_rate=learning_rate, **optimizer_kwargs)
        self.layers.append(DenseLayer(
            layers_sizes[-1], 2, activation='softmax',
            weights_initializer='glorot', optimizer=opt)
        )

    def forward(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        out = input
        for layer in self.layers:
            out = layer.feed_forward(out)
        return out

    def backward(self, d_Z: np.ndarray) -> np.ndarray:
        out = d_Z
        for layer in reversed(self.layers):
            out = layer.backward(out)
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
        return np.concatenate([p.flatten() for layer in self.layers for p in (layer.weights, layer.biases)])

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
