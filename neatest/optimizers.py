from typing import Union, List
from abc import ABC, abstractmethod

import numpy as np

from .connection import Connection, Weight

Array = Union[np.ndarray, np.generic]


class Optimizer(ABC):
    def __init__(self, weights: List[Weight], **kwargs):
        self.weights = weights

    def zero_grad(self) -> None:
        for i in range(len(self.weights)):
            self.weights[i].grad = 0.0

    @abstractmethod
    def step(self) -> None:
        ...


class Adam(Optimizer):
    def __init__(self, weights: List[Weight], lr: float, beta_1: float = 0.9, beta_2:
                 float = 0.999, epsilon: float = 1e-08):

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weights = weights

        self.m: Array = np.zeros(len(weights))
        self.v: Array = np.zeros(len(weights))

        self.t: int = 0

    def step(self) -> None:
        self.t += 1
        pad_size = len(self.weights) - self.m.size
        self.m = np.concatenate([self.m, np.zeros(pad_size)])
        self.v = np.concatenate([self.v, np.zeros(pad_size)])
        gradients: Array = np.array([weight.grad for weight in self.weights])
        lr = self.lr * (np.sqrt(1 - np.power(self.beta_2, self.t)) /
                        (1 - np.power(self.beta_1, self.t)))
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradients
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(gradients)
        step = -lr * self.m / (np.sqrt(self.v) + self.epsilon)
        assert step.size == len(self.weights)
        for i in range(len(self.weights)):
            self.weights[i].value += step[i]

