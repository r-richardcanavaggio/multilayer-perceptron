import numpy as np
from src.type import Tensor
from src.Math import sigmoid, softmax, glorot
from src.Optimizer import Optimizer, OptimizerFactory


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
            self.weights: Tensor = glorot(
                self.input_size, self.output_size)

        self.biases: Tensor = np.zeros(self.output_size)

    def __repr__(self) -> str:
        return (f"DenseLayer object: input = {self.input_size}"
                f" | output = {self.output_size}"
                f" | activation = {self.activation}"
                f" | weights initializer = {self.weights_initializer}")

    def feed_forward(self, input_matrix: Tensor) -> Tensor:
        self.input_matrix = input_matrix
        self.z = np.dot(self.input_matrix, self.weights) + self.biases
        self.activated_matrix: Tensor

        if self.activation == "sigmoid":
            self.activated_matrix = sigmoid(self.z)
        elif self.activation == "softmax":
            self.activated_matrix = softmax(self.z)

        return self.activated_matrix

    def _compute_activation_gradient(self, grad_output: Tensor) -> Tensor:
        if self.activation == "sigmoid":
            grad_output *= (self.activated_matrix *
                            (1 - self.activated_matrix))
        return grad_output

    def _compute_gradients(self, d_Z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        d_weights = (np.dot(self.input_matrix.transpose(), d_Z) /
                     self.input_matrix.shape[0])
        d_biases = d_Z.sum(axis=0) / self.input_matrix.shape[0]
        grad_input = np.dot(d_Z, self.weights.transpose())

        return d_weights, d_biases, grad_input

    def backward(self, grad_output: Tensor) -> Tensor:
        d_Z = self._compute_activation_gradient(grad_output)

        d_weights, d_biases, grad_input = self._compute_gradients(d_Z)

        self.weights, self.biases = self.optimizer.update(
            self.weights, self.biases, d_weights, d_biases)

        return grad_input

    @classmethod
    def from_computed(cls, w: Tensor, b: Tensor, act: str) -> DenseLayer:
        instance = cls(0, 0, act, '', OptimizerFactory.create('SGD', 0.0))
        instance.weights = w
        instance.biases = b
        return instance
