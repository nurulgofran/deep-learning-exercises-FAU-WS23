from generator import ImageGenerator
from matplotlib import pyplot as plt

if __name__ == '__main__':
	label_path = 'Labels.json'
	file_path = 'exercise_data/'
	row = 3
	column = 4
	gen = ImageGenerator(file_path, label_path, row * column, [32, 32, 3], rotation=False,
	                     mirroring=False, shuffle=False)
	images, labels = gen.next()
	# print(labels)
	# print(len(images))
	# plt.figure(figsize=(20, 4))
	index = 0
	f, axarr = plt.subplots(row, column)
	for i in range(row):
		for j in range(column):
			axarr[i, j].imshow(images[index], interpolation='nearest', vmax=0.1)
			index += 1
	plt.show()
