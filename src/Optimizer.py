from abc import ABC, abstractmethod
import numpy as np
from src.type import Tensor


class Optimizer(ABC):
    def __init__(self, learning_rate: float, **kwargs: float):
        self.lr = learning_rate

    @abstractmethod
    def update(self, weights: Tensor, biases: Tensor,
               d_weights: Tensor, d_biases: Tensor) -> tuple[Tensor, Tensor]:
        pass


OptimizerType = type[Optimizer]


class SGDOptimizer(Optimizer):
    def update(self, weights: Tensor, biases: Tensor,
               d_weights: Tensor, d_biases: Tensor) -> tuple[Tensor, Tensor]:
        new_weights = weights - self.lr * d_weights
        new_biases = biases - self.lr * d_biases
        return new_weights, new_biases


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate: float,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, **kwargs: float):
        super().__init__(learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m_w: Tensor | None = None
        self.v_w: Tensor | None = None
        self.m_b: Tensor | None = None
        self.v_b: Tensor | None = None

    def update(self, weights: Tensor, biases: Tensor,
               d_weights: Tensor, d_biases: Tensor) -> tuple[Tensor, Tensor]:
        if (self.m_w is None or self.v_w is None
           or self.m_b is None or self.v_b is None):
            self.m_w = np.zeros_like(weights)
            self.v_w = np.zeros_like(weights)
            self.m_b = np.zeros_like(biases)
            self.v_b = np.zeros_like(biases)

        self.t += 1

        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * d_weights
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (d_weights ** 2)

        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * d_biases
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (d_biases ** 2)

        m_hat_w = self.m_w / (1 - self.beta1 ** self.t)
        v_hat_w = self.v_w / (1 - self.beta2 ** self.t)

        m_hat_b = self.m_b / (1 - self.beta1 ** self.t)
        v_hat_b = self.v_b / (1 - self.beta2 ** self.t)

        new_weights = weights - (self.lr *
                                 (m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)))
        new_biases = biases - (self.lr *
                               (m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)))
        return new_weights, new_biases


class OptimizerFactory:
    _REGISTRY: dict[str, OptimizerType] = {
        'sgd': SGDOptimizer,
        'adam': AdamOptimizer
    }

    @classmethod
    def create(cls, name: str, learning_rate: float,
               **kwargs: float) -> Optimizer:
        name = name.lower()
        if name not in cls._REGISTRY:
            raise ValueError(f"Optimizer '{name}' not found."
                             f"Available: {list(cls._REGISTRY.keys())}")

        optimizer_class = cls._REGISTRY[name]
        return optimizer_class(learning_rate=learning_rate, **kwargs)
