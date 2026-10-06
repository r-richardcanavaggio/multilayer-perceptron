import numpy as np
from src.Math import sigmoid, softmax, glorot


class DenseLayer:
    def __init__(self, input_size: int,
                 output_size: int, activation: str,
                 weights_initializer: str):
        self.input_size = input_size
        self.output_size = output_size

        self.activation = activation
        self.weights_initializer = weights_initializer

        if self.weights_initializer == "glorot":
            self.weights = glorot(self.input_size, self.output_size)
        self.biases = np.zeros(self.output_size)

    def __repr__(self):
        return (f"DenseLayer object: input = {self.input_size}"
                f" | output = {self.output_size}"
                f" | activation = {self.activation}"
                f" | weights initializer = {self.weights_initializer}")

    def feed_forward(self, X) -> np.ndarray:
        self.X = X
        self.Z = np.dot(self.X, self.weights) + self.biases

        if self.activation == "sigmoid":
            self.A = sigmoid(self.Z)
        elif self.activation == "softmax":
            self.A = softmax(self.Z)

        return self.A

    def backward(self, d_Z_next, learning_rate) -> np.ndarray:
        if self.activation == "sigmoid":
            d_Z = d_Z_next * (self.A * (1 - self.A))
        elif self.activation == "softmax":
            d_Z = d_Z_next

        d_weights = np.dot(self.X.transpose(), d_Z) / self.X.shape[0]
        d_biases = d_Z.sum(axis=0) / self.X.shape[0]

        d_Prev = np.dot(d_Z, self.weights.transpose())

        self.weights = self.weights - (learning_rate * d_weights)
        self.biases = self.biases - (learning_rate * d_biases)

        return d_Prev

    @classmethod
    def from_computed(cls, weights: np.ndarray,
                      biases: np.ndarray, activation: str) -> DenseLayer:
        instance = cls(0, 0, activation, '')
        instance.weights = weights
        instance.biases = biases
        return instance
