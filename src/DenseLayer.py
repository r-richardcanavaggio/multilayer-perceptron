import numpy as np
from numpy.typing import NDArray
from src.Math import sigmoid, softmax, glorot


class DenseLayer:
    def __init__(self, input_size: int,
                 output_size: int, activation: str,
                 weights_initializer: str, optimizer: str):
        self.input_size = input_size
        self.output_size = output_size

        self.activation = activation
        self.weights_initializer = weights_initializer
        self.optimizer = optimizer

        if self.weights_initializer == "glorot":
            self.weights: NDArray[np.float64] = glorot(
                self.input_size, self.output_size
                )

        self.biases: NDArray[np.float64] = np.zeros(self.output_size)

        if self.optimizer == 'Adam':
            self.m_w: NDArray[np.float64] = np.zeros_like(self.weights)
            self.m_b: NDArray[np.float64] = np.zeros_like(self.biases)
            self.v_w: NDArray[np.float64] = np.zeros_like(self.weights)
            self.v_b: NDArray[np.float64] = np.zeros_like(self.biases)
            self.t: int = 1
            self.EPS: float = 1e-8

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

    def backward(self, d_Z_next: NDArray[np.float64],
                 learning_rate: float) -> NDArray[np.float64]:
        d_Z: NDArray[np.float64] = np.zeros_like(self.activated_matrix)

        if self.activation == "sigmoid":
            d_Z = d_Z_next * (
                self.activated_matrix * (1 - self.activated_matrix)
                )
        elif self.activation == "softmax":
            d_Z = d_Z_next

        d_weights = np.dot(
            self.input_matrix.transpose(), d_Z) / self.input_matrix.shape[0]
        d_biases = d_Z.sum(axis=0) / self.input_matrix.shape[0]

        d_Prev = np.dot(d_Z, self.weights.transpose())

        if self.optimizer == 'SGD':
            self.weights = self.weights - (learning_rate * d_weights)
            self.biases = self.biases - (learning_rate * d_biases)
        elif self.optimizer == 'Adam':
            self.m_w = 0.9 * self.m_w + 0.1 * d_weights
            self.v_w = 0.999 * self.v_w + 0.001 * (d_weights ** 2)

            self.m_b = 0.9 * self.m_b + 0.1 * d_biases
            self.v_b = 0.999 * self.v_b + 0.001 * (d_biases ** 2)

            m_hat_w = self.m_w / (1 - 0.9 ** self.t)
            v_hat_w = self.v_w / (1 - 0.999 ** self.t)

            m_hat_b = self.m_b / (1 - 0.9 ** self.t)
            v_hat_b = self.v_b / (1 - 0.999 ** self.t)

            self.t += 1
            self.weights -= (learning_rate *
                             (m_hat_w / (np.sqrt(v_hat_w) + self.EPS)))
            self.biases -= (learning_rate *
                            (m_hat_b / (np.sqrt(v_hat_b) + self.EPS)))

        return d_Prev

    @classmethod
    def from_computed(cls, weights: np.ndarray,
                      biases: np.ndarray, activation: str) -> DenseLayer:
        instance = cls(0, 0, activation, '', '')
        instance.weights = weights
        instance.biases = biases
        return instance
