import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exponents = np.exp(x)

    sum_of_exponents = np.sum(exponents, axis=1, keepdims=True)

    probabilities = np.exp(x) / sum_of_exponents
    return probabilities


def glorot(row: int, col: int):
    n = row * col
    lower, upper = -(1.0 / np.sqrt(n)), (1.0 / np.sqrt(n))

    numbers = np.random.rand(n)
    scaled = lower + numbers * (upper - lower)
    scaled = scaled.reshape(row, col)
    return scaled


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray):
    bce = -np.mean(
        y_true * np.log(y_pred[:, 0]) + (1 - y_true) * np.log(y_pred[:, 1])
    )
    return bce
