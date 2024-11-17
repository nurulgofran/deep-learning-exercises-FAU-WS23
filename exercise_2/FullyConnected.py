import numpy as np

from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self._optimizer = None
        self.input_tensor = []
        self.trainable = True
        self.gradient_weights = None

        self.weights = np.random.rand(in_size, out_size)
        self.weights_bias = np.ones((1, out_size))
        self.weights = np.concatenate((self.weights, self.weights_bias), axis=0)

    def forward(self, in_tensor):
        in_bias = np.ones((in_tensor.shape[0], 1))
        self.input_tensor = in_tensor
        self.input_tensor = np.concatenate((in_tensor, in_bias), axis=1)
        return np.dot(self.input_tensor, self.weights)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def backward(self, error_tensor):
        # gradient with respect to X
        gradient_x = np.dot(error_tensor, np.transpose(np.delete(self.weights, self.weights.shape[0] - 1, axis=0)))

        # gradient with respect to W
        self.gradient_weights = np.dot(np.transpose(self.input_tensor), error_tensor)
        if self.optimizer != None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return gradient_x

    def initialize(self, weights_initializer, bias_initializer):
        # self.weights = np.delete(self.weights, self.weights.shape[0] - 1, axis=0)
        self.weights = weights_initializer.initialize((self.in_size, self.out_size), self.in_size, self.out_size)
        bias = bias_initializer.initialize(self.weights_bias.shape, self.in_size, self.out_size)
        self.weights = np.concatenate((self.weights, bias), axis=0)