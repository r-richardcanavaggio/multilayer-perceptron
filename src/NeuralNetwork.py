import numpy as np


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
        np.savez('model_weights.npz', *self.layers, allow_pickle=True)

    @classmethod
    def from_npz(cls, filename: str):
        loaded_layers = np.load(filename)
        return cls(list(loaded_layers.values()), None)
