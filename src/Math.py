import numpy as np
from src.type import Tensor


def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + np.exp(-x))


def softmax(x: Tensor) -> Tensor:
    exponents = np.exp(x)

    sum_of_exponents = np.sum(exponents, axis=1, keepdims=True)

    probabilities = np.exp(x) / sum_of_exponents
    return probabilities


def glorot(row: int, col: int) -> Tensor:
    n = row * col
    lower, upper = -(1.0 / np.sqrt(n)), (1.0 / np.sqrt(n))

    numbers = np.random.rand(n)
    scaled = lower + numbers * (upper - lower)
    scaled = scaled.reshape(row, col)
    return scaled


def binary_cross_entropy(y_true: Tensor, y_pred: Tensor) -> float:
    bce = -np.mean(
        y_true * np.log(y_pred[:, 1]) + (1 - y_true) * np.log(y_pred[:, 0])
    )
    return bce


def compute_accuracy(y_true: Tensor, y_pred: Tensor) -> float:
    raw_predict = y_pred.copy()
    prediction = np.argmax(raw_predict, axis=1)
    return (y_true == prediction).sum() / y_pred.shape[0]
