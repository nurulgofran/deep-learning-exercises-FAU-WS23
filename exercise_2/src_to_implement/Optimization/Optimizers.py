import numpy as np


class Sgd:
    learning_rate = 0.

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # print(weight_tensor - self.learning_rate * gradient)
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:

    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.previous_v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        v = self.momentum_rate * self.previous_v - self.learning_rate * gradient_tensor
        self.previous_v = v
        updated_weight = weight_tensor + v
        return updated_weight


class Adam:

    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.previous_v = 0
        self.previous_r = 0
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.k += 1
        g = gradient_tensor
        v = self.mu * self.previous_v + (1 - self.mu) * g
        r = self.rho * self.previous_r + (1 - self.rho) * (g * g)
        self.previous_v = v
        self.previous_r = r
        v = v / (1 - self.mu ** self.k)
        r = r / (1 - self.rho ** self.k)
        return weight_tensor - self.learning_rate * (v / (np.sqrt(r) + np.finfo(float).eps))
