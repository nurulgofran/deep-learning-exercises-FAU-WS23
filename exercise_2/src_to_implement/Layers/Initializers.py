import numpy as np


class Constant:

    def __init__(self, value=0.1):
        self.default_value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.default_value)


class UniformRandom:

    def __init__(self):
        self.low = 0
        self.high = 1

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(self.low, self.high, size=weights_shape)


class Xavier:

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, sigma, size=weights_shape)


class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(0, sigma, size=weights_shape)
