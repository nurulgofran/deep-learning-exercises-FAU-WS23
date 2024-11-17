import numpy as np

from Layers.Base import BaseLayer


class Pooling(BaseLayer):
	def __init__(self, stride_shape, pooling_shape):
		super().__init__()
		self.stride_shape = stride_shape
		self.pooling_shape = pooling_shape
		self.indices = []
		self.input_tensor = None

	def forward(self, input_tensor):
		self.input_tensor = input_tensor
		output_tensor = np.empty(self.output_shape(input_tensor))

		for batch in range(input_tensor.shape[0]):
			for channel in range(input_tensor.shape[1]):
				for i in range(0, output_tensor.shape[2]):
					for j in range(0, output_tensor.shape[3]):
						pool_array = input_tensor[batch][channel][
						             i * self.stride_shape[0]:i * self.stride_shape[0] + self.pooling_shape[0],
						             j * self.stride_shape[1]:j * self.stride_shape[1] + self.pooling_shape[1]]
						output_tensor[batch][channel][i][j] = np.amax(pool_array)

						index = np.unravel_index(np.argmax(pool_array), pool_array.shape)
						self.indices.append((i * self.stride_shape[0] + index[0], j * self.stride_shape[1] + index[1]))
		return output_tensor

	def output_shape(self, input_tensor):
		input_shape = input_tensor[0][0].shape

		x = int((input_shape[0] - self.pooling_shape[0]) / self.stride_shape[0]) + 1
		y = int((input_shape[1] - self.pooling_shape[1]) / self.stride_shape[1]) + 1
		output_shape = input_tensor.shape[0], input_tensor.shape[1], x, y

		return output_shape

	def backward(self, error_tensor):
		output_error = np.zeros(self.input_tensor.shape)
		k = 0
		for batch in range(error_tensor.shape[0]):
			for channel in range(error_tensor.shape[1]):
				for i in range(error_tensor.shape[2]):
					for j in range(error_tensor.shape[3]):
						output_error[batch][channel][self.indices[k]] += error_tensor[batch][channel][i][j]
						k += 1
		return output_error
