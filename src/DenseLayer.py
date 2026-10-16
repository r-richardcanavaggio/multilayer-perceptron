import numpy as np
from numpy.typing import NDArray
from src.Math import sigmoid, softmax, glorot
from src.Optimizer import Optimizer


class DenseLayer:
    def __init__(self, input_size: int,
                 output_size: int, activation: str,
                 weights_initializer: str, optimizer: Optimizer):
        self.input_size = input_size
        self.output_size = output_size

        self.activation = activation
        self.weights_initializer = weights_initializer
        self.optimizer = optimizer

        if self.weights_initializer == "glorot":
            self.weights: NDArray[np.float64] = glorot(
                self.input_size, self.output_size)

        self.biases: NDArray[np.float64] = np.zeros(self.output_size)

    def __repr__(self):
        return (f"DenseLayer object: input = {self.input_size}"
                f" | output = {self.output_size}"
                f" | activation = {self.activation}"
                f" | weights initializer = {self.weights_initializer}")

    def feed_forward(self,
                     input_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        self.input_matrix = input_matrix
        self.z = np.dot(self.input_matrix, self.weights) + self.biases
        self.activated_matrix: NDArray[np.float64]

        if self.activation == "sigmoid":
            self.activated_matrix = sigmoid(self.z)
        elif self.activation == "softmax":
            self.activated_matrix = softmax(self.z)

        return self.activated_matrix

    def backward(self, d_Z_next: NDArray[np.float64]) -> NDArray[np.float64]:
        d_Z: NDArray[np.float64] = np.zeros_like(self.activated_matrix)

        if self.activation == "sigmoid":
            d_Z = d_Z_next * (self.activated_matrix *
                              (1 - self.activated_matrix))
        elif self.activation == "softmax":
            d_Z = d_Z_next

        d_weights = (np.dot(self.input_matrix.transpose(), d_Z) /
                     self.input_matrix.shape[0])
        d_biases = d_Z.sum(axis=0) / self.input_matrix.shape[0]

        d_Prev = np.dot(d_Z, self.weights.transpose())

        self.weights, self.biases = self.optimizer.update(
            self.weights, self.biases, d_weights, d_biases)

        return d_Prev

    @classmethod
    def from_computed(cls, weights: np.ndarray,
                      biases: np.ndarray, activation: str) -> DenseLayer:
        instance = cls(0, 0, activation, '', '')
        instance.weights = weights
        instance.biases = biases
        return instance
