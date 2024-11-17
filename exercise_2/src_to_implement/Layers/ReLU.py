import numpy as np
import Layers.Base


class ReLU(Layers.Base.BaseLayer):

	def __init__(self):
		super().__init__()
		self.input_tensor = None

	def forward(self, input_tensor):
		self.input_tensor = input_tensor
		return np.maximum(input_tensor, 0)

	def backward(self, error_tensor):
		grad = self.input_tensor > 0
		error_tensor *= grad
		return error_tensor
