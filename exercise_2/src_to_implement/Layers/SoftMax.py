import numpy as np
import Layers.Base


class SoftMax(Layers.Base.BaseLayer):

	def __init__(self):
		super().__init__()
		self.input_tensor = None
		self.output_tensor = None

	def forward(self, input_tensor):
		self.input_tensor = input_tensor
		# to keep every value below 1
		x_k_updated = input_tensor - np.max(input_tensor)
		exp_ = np.exp(x_k_updated)
		self.output_tensor = exp_ / np.sum(exp_, axis=1, keepdims=True)
		return self.output_tensor

	def backward(self, error_tensor):
		e_hat = self.output_tensor * (error_tensor - np.sum(error_tensor * self.output_tensor, axis=1, keepdims=True))
		return e_hat
