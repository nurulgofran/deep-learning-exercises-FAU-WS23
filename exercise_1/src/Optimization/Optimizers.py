class Sgd:
	learning_rate = 0.

	def __init__(self, learning_rate: float):
		self.learning_rate = learning_rate

	def calculate_update(self, weight_tensor, gradient_tensor):
		# print(weight_tensor - self.learning_rate * gradient)
		return weight_tensor - self.learning_rate * gradient_tensor
