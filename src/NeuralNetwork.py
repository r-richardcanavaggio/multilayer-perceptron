import numpy as np
from src.DenseLayer import DenseLayer


class NeuralNetwork:
    def __init__(self, layers: list, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

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

    def export(self):
        nn_dict = {}
        for i, layer in enumerate(self.layers):
            nn_dict[f"W_{i}"] = layer.weights
            nn_dict[f"b_{i}"] = layer.biases
            nn_dict[f"A_{i}"] = layer.activation

        print(f"ss{nn_dict}")
        np.savez('data/model_weights.npz', **nn_dict)

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
