import matplotlib.pyplot as plt
import numpy as np


class Checker:
	def __init__(self, resolution, tile_size):
		self.resolution = resolution
		self.tile = tile_size

		# output holds output array
		self.output = None

	def draw(self):
		dim = self.resolution // (self.tile * 2)

		tile_shape = (self.tile, self.tile)
		tile_block = np.concatenate((
			np.concatenate((np.zeros(shape=tile_shape, dtype=int), np.ones(shape=tile_shape, dtype=int)), axis=0),
			np.concatenate((np.ones(shape=tile_shape, dtype=int), np.zeros(shape=tile_shape, dtype=int)), axis=0)
		), axis=1)

		board = np.tile(tile_block, (dim, dim))

		self.output = board

		return board.copy()

	def show(self):
		plt.imshow(self.output, cmap="gray")
		plt.show()


class Circle:
	def __init__(self, resolution, radius, position):
		self.resolution = resolution
		self.radius = radius
		self.position = position

		# output holds output array
		self.output = None

	def draw(self):
		x = np.linspace(-0, self.resolution, self.resolution, dtype=int)
		y = np.linspace(-0, self.resolution, self.resolution, dtype=int)
		# create coordinate matrix
		xx, yy = np.meshgrid(x, y)

		h, w = self.position

		# print((xx-h)** 2 + (yy-w)** 2)

		# finding the position in array and to make a bool array
		circle = np.array(((xx - h) ** 2 + (yy - w) ** 2) <= self.radius ** 2)
		self.output = circle
		return circle.copy()

	def show(self):
		plt.imshow(self.output, cmap="gray")
		plt.show()


class Spectrum:
	def __init__(self, resolution):
		self.resolution = resolution

	def draw(self):
		sprectrum = np.zeros([self.resolution, self.resolution, 3])

		sprectrum[:, :, 0] = np.linspace(0, 1, self.resolution)
		sprectrum[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)
		sprectrum[:, :, 2] = np.linspace(1, 0, self.resolution)

		self.output = sprectrum

		return self.output.copy()

	def show(self):
		plt.imshow(self.output)
		plt.show()


if __name__ == "__main__":
	c = Spectrum(255)
	# c = Checker(128, 4)
	# c = Circle(1024, 64, (250, 500))
	c.draw()
	c.show()
