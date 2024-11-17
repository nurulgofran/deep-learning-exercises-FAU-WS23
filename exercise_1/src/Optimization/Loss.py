import numpy as np


class CrossEntropyLoss:

	def __init__(self):
		self.prediction_tensor = None
		self.loss = None

	def forward(self, prediction_tensor, label_tensor):
		self.prediction_tensor = prediction_tensor
		epsilon = np.finfo(float).eps
		loss = -np.sum(label_tensor * np.log(prediction_tensor + epsilon))
		return loss

	def backward(self, label_tensor):
		epsilon = np.finfo(float).eps
		return - label_tensor / (self.prediction_tensor + epsilon)
