import numpy as np
from numpy.typing import NDArray
from typing import Any


def sigmoid(x: Any):
    return 1 / (1 + np.exp(-x))


def softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    exponents = np.exp(x)

    sum_of_exponents = np.sum(exponents, axis=1, keepdims=True)

    probabilities = np.exp(x) / sum_of_exponents
    return probabilities


def glorot(row: int, col: int) -> NDArray[np.float64]:
    n = row * col
    lower, upper = -(1.0 / np.sqrt(n)), (1.0 / np.sqrt(n))

    numbers = np.random.rand(n)
    scaled = lower + numbers * (upper - lower)
    scaled = scaled.reshape(row, col)
    return scaled


def binary_cross_entropy(y_true: NDArray[np.float64],
                         y_pred: NDArray[np.float64]) -> float:
    bce = -np.mean(
        y_true * np.log(y_pred[:, 1]) + (1 - y_true) * np.log(y_pred[:, 0])
    )
    return bce


def compute_accuracy(y_true: NDArray[np.float64],
                     y_pred: NDArray[np.float64]) -> float:
    raw_predict = y_pred.copy()
    prediction = np.argmax(raw_predict, axis=1)
    return (y_true == prediction).sum() / y_pred.shape[0]
