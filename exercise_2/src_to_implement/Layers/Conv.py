import copy

import scipy.signal

from Layers.Base import BaseLayer
from Layers.Initializers import *


class Conv(BaseLayer):
	
	def __init__(self, stride_shape, convolution_shape, num_kernels):
		"""

		:param stride_shape: single value or tuple(1 for next row), for tuple (provides strides for row switching)
		:param convolution_shape: {1D, 2D} for 1D, i.e. [c, m]. For 2D i.e. [c, m, n]. c= channel
		:param num_kernels: integer. Number of filter/kernels
		"""
		super().__init__()
		self.trainable = True
		self.input_tensor = None
		self.stride_shape = stride_shape
		self.convolution_shape = convolution_shape
		self.num_kernels = num_kernels
		self.gradient_weights = None
		self.gradient_bias = None
		self.bias_optimizer = None
		self.grad_optimizer = None
		
		weight_shape = list(convolution_shape)
		weight_shape.insert(0, num_kernels)
		weight_shape = tuple(weight_shape)
		self.weights = UniformRandom().initialize(weight_shape, 0, 0)
		self.bias = UniformRandom().initialize(self.num_kernels, 0, 0)
	
	@property
	def optimizer(self):
		return self.grad_optimizer

	@optimizer.setter
	def optimizer(self, optimizer):
		self.grad_optimizer = optimizer
		self.bias_optimizer = copy.deepcopy(optimizer)
	
	def forward(self, input_tensor):
		self.input_tensor = input_tensor
		output_tensor = np.empty(self.output_shape(input_tensor))
		for batch in range(len(input_tensor)):
			for kernel in range(self.num_kernels):
				correlated_tensor = None
				for channel in range(len(self.weights[kernel])):
					tensor = scipy.signal.correlate(input_tensor[batch][channel],
					                                self.weights[kernel][channel],
					                                mode="same")
					if correlated_tensor is None:
						correlated_tensor = tensor
					else:
						correlated_tensor += tensor
				if correlated_tensor.ndim == 1:
					output_tensor[batch][kernel] = correlated_tensor[::self.stride_shape[0]]
				elif correlated_tensor.ndim == 2:
					output_tensor[batch][kernel] = correlated_tensor[::self.stride_shape[0], ::self.stride_shape[1]]
				# add bias to output per filter
				output_tensor[batch][kernel] += self.bias[kernel]
		
		return output_tensor
	
	def output_shape(self, input_tensor):
		input_shape = input_tensor[0][0].shape
		input_dim = input_tensor[0][0].ndim
		output_tensor_shape = None
		
		if input_dim == 1:
			x = int((input_shape[0] - self.convolution_shape[1] + sum(self.padding(self.convolution_shape[1]))) /
			        self.stride_shape[0]) + 1
			output_tensor_shape = input_tensor.shape[0], self.num_kernels, x
		elif input_dim == 2:
			x = int((input_shape[0] - self.convolution_shape[1] + sum(self.padding(self.convolution_shape[1]))) /
			        self.stride_shape[0]) + 1
			y = int((input_shape[1] - self.convolution_shape[2] + sum(self.padding(self.convolution_shape[2]))) /
			        self.stride_shape[1]) + 1
			output_tensor_shape = input_tensor.shape[0], self.num_kernels, x, y
		
		return output_tensor_shape
	
	def padding(self, size):
		padding1 = padding2 = int(size / 2)
		if size % 2 == 0:
			padding2 -= 1
		return padding1, padding2
	
	def initialize(self, weights_initializer, bias_initializer):
		self.weights = weights_initializer.initialize(self.weights.shape,
		                                              np.prod(self.convolution_shape),
		                                              np.prod(self.convolution_shape[1:]) * self.num_kernels)
		self.bias = bias_initializer.initialize(self.bias.shape,
		                                        np.prod(self.convolution_shape),
		                                        np.prod(self.convolution_shape[1:]) * self.num_kernels)
	
	def backward(self, error_tensor):
		# in back pass, error channel converts to filter and filter converts to channel???
		# gradient with respect to bias
		self.gradient_bias = np.zeros(self.bias.shape)
		
		for batch in range(len(error_tensor)):
			for channel in range(len(error_tensor[0])):
				self.gradient_bias[channel] += np.sum(error_tensor[batch][channel])
		
		# gradient with respect to output
		# this is what goes back to previous layer,
		# so it should be the shape of input_tensor
		output_tensor = np.zeros(self.input_tensor.shape)
		transformed_weights = self.transform_weights()
		
		for batch in range(len(error_tensor)):
			for kernel in range(self.num_kernels):
				# result is 1d/2d
				result = np.zeros(self.input_tensor[batch][0].shape)
				if error_tensor[batch][kernel].ndim == 1:
					result[::self.stride_shape[0]] = error_tensor[batch][kernel]
				elif error_tensor[batch][kernel].ndim == 2:
					result[::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[batch][kernel]
				
				for channel in range(len(transformed_weights)):
					output_tensor[batch][channel] += scipy.signal.convolve(result, transformed_weights[channel][kernel],
					                                                       mode="same")
		
		# gradient with respect to weights
		self.gradient_weights = np.zeros(self.weights.shape)
		
		for batch in range(len(error_tensor)):
			for kernel in range(self.num_kernels):
				for input_channel in range(len(self.input_tensor[0])):
					temp_tensor = None
					channel_error = np.zeros(self.input_tensor[batch][input_channel].shape)
					
					if self.weights[0][0].ndim == 1:
						temp_tensor = np.pad(self.input_tensor[batch][input_channel],
						                     self.padding(self.convolution_shape[1]),
						                     mode='constant')
						channel_error[::self.stride_shape[0]] = error_tensor[batch][kernel]
					
					elif self.weights[0][0].ndim == 2:
						temp_tensor = np.pad(self.input_tensor[batch][input_channel],
						                     (self.padding(self.convolution_shape[1]),
						                      self.padding(self.convolution_shape[2])),
						                     mode='constant')
						channel_error[::self.stride_shape[0], ::self.stride_shape[1]] \
							= error_tensor[batch][kernel]
					
					if temp_tensor is not None:
						tensor = scipy.signal.correlate(temp_tensor, channel_error, mode='valid')
						self.gradient_weights[kernel][input_channel] += tensor
		
		if self.grad_optimizer is not None:
			self.weights = self.grad_optimizer.calculate_update(self.weights, self.gradient_weights)
			self.bias = self.bias_optimizer.calculate_update(self.bias, self.gradient_bias)
		
		return output_tensor
	
	def transform_weights(self):
		transformed_weights = np.empty(self.weights.shape)
		transformed_weights = np.moveaxis(transformed_weights, 0, 1)
		for kernel in range(self.num_kernels):
			for channel in range(len(self.weights[kernel])):
				transformed_weights[channel][kernel] = self.weights[kernel][channel]
		return transformed_weights
