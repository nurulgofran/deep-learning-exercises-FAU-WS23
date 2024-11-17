import numpy as np

import Layers.Base


class FullyConnected(Layers.Base.BaseLayer):

	def __init__(self, input_size, output_size):
		super().__init__()
		self.input_tensor = None
		self.output_tensor = None
		self.optimizer = None
		self.trainable = True
		self.weights = np.random.uniform(0, 1, [input_size + 1, output_size])

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
