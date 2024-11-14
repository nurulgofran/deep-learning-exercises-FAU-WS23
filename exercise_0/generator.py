import json
import os.path

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.ndimage


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
	def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
		# Define all members of your generator class object as global members here.
		# These need to include:
		# the batch size
		# the image size
		# flags for different augmentations and whether the data should be shuffled for each epoch
		# Also depending on the size of your data-set you can consider loading all images into memory here already.
		# The labels are stored in json format and can be directly loaded as dictionary.
		# Note that the file names correspond to the dicts of the label dictionary.
		self.file_path = file_path
		self.label_path = label_path
		self.batch_size = batch_size
		# self.height, self.width, self.channel = image_size
		self.image_size = image_size
		self.rotation = rotation
		self.mirroring = mirroring
		self.shuffle = shuffle

		# first epoch is 0
		self.epoch = 0
		# first pointer for reading data
		self.read_ptr = 0

		# load json data
		self.json_data = list(json.load(open(label_path)).items())

		# add shuffle flag
		if self.shuffle:
			# shuffle json list
			np.random.shuffle(self.json_data)

		self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
		                   7: 'horse', 8: 'ship', 9: 'truck'}

	def next(self):
		# This function creates a batch of images and corresponding labels and returns them.
		# In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
		# Note that your amount of total data might not be divisible without remainder with the batch_size.
		# Think about how to handle such cases
		# TODO: implement next method

		self.mirror_checker = True
		self.rotation_checker = True

		# try to load data from json data of batch size
		images = []
		labels = []
		for _ in range(self.batch_size):
			# check if epoch is reached or not
			if self.read_ptr == len(self.json_data):
				self.read_ptr = 0
				self.epoch += 1
				# if shuffle true then do
				if self.shuffle:
					np.random.shuffle(self.json_data)

			# just read the file
			file_name = f"{self.json_data[self.read_ptr][0]}.npy"
			image_path = os.path.join(self.file_path, file_name)
			import skimage.transform as skt
			# image = np.resize(np.load(image_path), self.image_size)
			image = skt.resize(np.load(image_path), self.image_size)
			images.append(self.augment(image))
			labels.append(self.json_data[self.read_ptr][1])

			# increment pointer
			self.read_ptr += 1

		return np.array(images), np.array(labels)

	def augment(self, img):
		# this function takes a single image as an input and performs a random transformation
		# (mirroring and/or rotation) on it and outputs the transformed image
		# TODO: implement augmentation function

		# check for mirror
		if self.mirroring:
			# create a mirror image of that image
			if self.mirror_checker:
				img = np.flipud(img)
				self.mirror_checker = False
			else:
				self.mirror_checker = True

		# check if rotation
		if self.rotation:
			# create a mirror image of that image
			if self.rotation_checker:
				img = scipy.ndimage.rotate(img, np.random.choice([90, 180, 270]))
				self.rotation_checker = False
			else:
				self.rotation_checker = True

		return img

	def current_epoch(self):
		# return the current epoch number
		return self.epoch

	def class_name(self, x):
		# This function returns the class name for a specific input
		# TODO: implement class name function
		return self.class_dict[int(x)]

	def show(self):
		# In order to verify that the generator creates batches as required, this functions calls next to get a
		# batch of images and labels and visualizes it.
		# TODO: implement show method
		images, labels = self.next()
		row = self.batch_size // 6
		col = 6
		plt.figure()
		img_num = 1
		for image, label in zip(images, labels):
			# fig.add_subplot(row, col, img_num)
			plt.subplot(row, col, img_num)
			plt.title(self.class_name(label))
			plt.axis('off')
			plt.imshow(image)
			img_num += 1

		plt.tight_layout()
		plt.show()


if __name__ == '__main__':
	gen = ImageGenerator('exercise_data/', 'Labels.json', 24, [84, 84, 3], rotation=False, mirroring=False,
	                     shuffle=False)
	gen.show()
