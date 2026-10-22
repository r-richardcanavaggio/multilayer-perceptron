import numpy as np
from src.DenseLayer import DenseLayer
from src.Optimizer import OptimizerFactory
from src.type import Tensor
from src.StandardScaler import StandardScaler


class NeuralNetwork:
    def __init__(self, input_size: int,
                 layers_sizes: list[int], learning_rate: float | None,
                 activation: str, optimizer: str, **optimizer_kwargs: float):
        self.layers: list[DenseLayer] = []
        self.optimizer_name: str = optimizer
        self.scaler: StandardScaler | None = None

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

    def forward(self, input: Tensor) -> Tensor:
        out = input
        for layer in self.layers:
            out = layer.feed_forward(out)
        return out

    def backward(self, d_Z: Tensor) -> Tensor:
        out = d_Z
        for layer in reversed(self.layers):
            out = layer.backward(out)
        return out

    def update(self) -> None:
        for layer in self.layers:
            layer.update()

    def export(self, path: str, scaler: StandardScaler | None = None) -> None:
        nn_dict: dict[str, Tensor | str] = {}

        for i, layer in enumerate(self.layers):
            nn_dict[f"W_{i}"] = layer.weights
            nn_dict[f"b_{i}"] = layer.biases
            nn_dict[f"A_{i}"] = layer.activation

        if scaler is not None:
            nn_dict["scaler_mean"] = scaler.mean
            nn_dict["scaler_std"] = scaler.std

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
    def from_npz(cls, filename: str) -> NeuralNetwork:
        loaded_layers = np.load(filename)
        layers: list[DenseLayer] = []

        num_layers = sum(1 for k in loaded_layers.files if k.startswith("W_"))

        for i in range(num_layers):
            weights = loaded_layers[f"W_{i}"]
            biases = loaded_layers[f"b_{i}"]
            activation = loaded_layers[f"A_{i}"]
            layer = DenseLayer.from_computed(weights, biases, activation)
            layers.append(layer)

        instance = cls(0, [], 0, "", "")
        instance.layers = layers

        if "scaler_mean" in loaded_layers and "scaler_std" in loaded_layers:
            scaler = StandardScaler()
            scaler.mean = loaded_layers["scaler_mean"]
            scaler.std = loaded_layers["scaler_std"]
            instance.scaler = scaler
        else:
            instance.scaler = None

        return instance

    def get_weights_vector(self) -> Tensor:
        flat = [p.flatten() for layer in self.layers
                for p in (layer.weights, layer.biases)]
        return np.concatenate(flat)

    def set_weights_vector(self, theta: Tensor) -> None:
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

    def size(self) -> list[int]:
        size: list[int] = []

        for layer in self.layers:
            size.append(layer.input_size)
        return size
