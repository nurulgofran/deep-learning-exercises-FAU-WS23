from Layers.Base import BaseLayer


class Flatten(BaseLayer):

	def __init__(self):
		super().__init__()
		self.trainable = False
		self.input_tensor = None

	def forward(self, input_tensor):
		self.input_tensor = input_tensor
		return input_tensor.reshape(len(input_tensor), input_tensor[0].flatten().size)

	def backward(self, error_tensor):
		return error_tensor.reshape(self.input_tensor.shape)
