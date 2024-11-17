import numpy as np


# constant init
class Constant:
    def __init__(self, constant_value = 0.1):
        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.constant_value)


# uniformly random selection [0,1) including 0
class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0, 1, weights_shape)


# Xavier zero means gaussian dist to Fi
class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(0, np.sqrt(2 / (fan_in + fan_out)), weights_shape)


# improvement of Xavier
class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(0, np.sqrt(2 / fan_in), weights_shape)
