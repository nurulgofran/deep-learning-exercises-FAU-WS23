import copy

import Layers.Base


class NeuralNetwork:

	def __init__(self, optimizers):
		self._batch_data = None
		self.optimizer = optimizers
		self.layers = list()
		self.loss = list()
		self.data_layer = None
		self.loss_layer = None

	def forward(self):
		self._batch_data = self.data_layer.next()
		input_tensor, label_tensor = self._batch_data
		for layer in self.layers:
			input_tensor = layer.forward(input_tensor)
		loss = self.loss_layer.forward(input_tensor, label_tensor)
		# print(loss)
		return loss

	def backward(self):
		_, label_tensor = self._batch_data
		error_tensor = self.loss_layer.backward(label_tensor)
		for layer in reversed(self.layers):
			# print(f'layer {layer}')
			error_tensor = layer.backward(error_tensor)
		return error_tensor

	def append_layer(self, layer: Layers.Base.BaseLayer):
		if layer.trainable:
			layer.optimizer = copy.deepcopy(self.optimizer)
		self.layers.append(layer)

	def train(self, iteration: int):
		for i in range(iteration):
			loss = self.forward()
			self.loss.append(loss)
			self.backward()
		return

	def test(self, input_tensor):
		for layer in self.layers:
			input_tensor = layer.forward(input_tensor)
		return input_tensor
