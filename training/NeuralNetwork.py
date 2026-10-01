import numpy as np


class NeuralNetwork:
    def __init__(self, layers, learning_rate):
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
