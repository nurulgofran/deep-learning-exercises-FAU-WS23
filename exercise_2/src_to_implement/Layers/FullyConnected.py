import numpy as np

import Layers.Base


class FullyConnected(Layers.Base.BaseLayer):

	def __init__(self, input_size, output_size):
		super().__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.input_tensor = None
		self.output_tensor = None
		self.optimizer = None
		self.trainable = True
		self.weights = np.random.rand(input_size, output_size)
		self.weights_bias = np.ones((1, self.output_size))
		self.weights = np.concatenate((self.weights, self.weights_bias), axis=0)

	def forward(self, input_tensor):
		# print(input_tensor.shape)
		self.input_tensor = np.hstack((input_tensor, np.ones((input_tensor.shape[0], 1))))
		self.output_tensor = np.matmul(self.input_tensor, self.weights)
		return self.output_tensor

	def backward(self, error_tensor):
		# print(error_tensor.shape)
		# gradient with respect to weight
		self.gradient_weights = np.matmul(self.input_tensor.transpose(), error_tensor)
		if self.optimizer is not None:
			self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
		# gradient with respect to input
		gradient_x = np.matmul(error_tensor, self.weights.transpose())
		return gradient_x[:, :-1]

	@property
	def optimizer(self):
		return self._optimizer

	@optimizer.setter
	def optimizer(self, value):
		self._optimizer = value
		
	def initialize(self, weights_initializer, bias_initializer):
		# self.weights = np.delete(self.weights, self.weights.shape[0] - 1, axis=0)
		self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
		bias = bias_initializer.initialize(self.weights_bias.shape, self.input_size, self.output_size)
		self.weights = np.concatenate((self.weights, bias), axis=0)
