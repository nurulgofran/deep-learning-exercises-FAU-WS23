from copy import deepcopy

import numpy as np
import scipy.ndimage
import scipy.signal

import Layers
from Layers.Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self._optimizer = None
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.input_tensor = None
        self.gradient_weights = None
        self.gradient_bias = None

        weight_shape = list(convolution_shape)
        weight_shape.insert(0, num_kernels)
        weight_shape = tuple(weight_shape)
        self.weights = Layers.Initializers.UniformRandom().initialize(weight_shape, 0, 0)
        self.bias = Layers.Initializers.UniformRandom().initialize(self.num_kernels, 0, 0)
        self._bias_optimizer = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._bias_optimizer = deepcopy(optimizer)

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
        pad1 = pad2 = int(size / 2)
        if size % 2 == 0:
            pad2 -= 1
        return pad1, pad2

    def backward(self, error_tensor):
        # gradient bias

        self.gradient_bias = np.zeros(self.bias.shape)
        for batch in range(error_tensor.shape[0]):
            for channel in range(error_tensor.shape[1]):
                self.gradient_bias[channel] += np.sum(error_tensor[batch][channel])

        # gradient with respect to lower layer
        output_tensor = np.zeros(self.input_tensor.shape)
        rearranged_weights = self.rearrange_weights()
        for batch in range(len(error_tensor)):
            for error_channel in range(self.num_kernels):
                result = np.zeros(self.input_tensor[batch][0].shape)
                if error_tensor[batch][error_channel].ndim == 1:
                    result[::self.stride_shape[0]] = error_tensor[batch][error_channel]
                elif error_tensor[batch][error_channel].ndim == 2:
                    result[::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[batch][error_channel]

                for kernel in range(rearranged_weights.shape[0]):
                    output_tensor[batch][kernel] += scipy.signal.convolve(result,
                                                                          rearranged_weights[kernel][error_channel],
                                                                          mode="same")

        # gradient with respect to weights
        self.gradient_weights = np.zeros(self.weights.shape)

        for batch in range(error_tensor.shape[0]):
            for error_channel in range(self.num_kernels):
                for input_channel in range(self.input_tensor.shape[1]):
                    input_tensor = None
                    error = np.zeros(self.input_tensor[batch][input_channel].shape)
                    if self.weights[0][0].ndim == 1:
                        input_tensor = np.pad(self.input_tensor[batch][input_channel],
                                              self.padding(self.convolution_shape[1]),
                                              mode="constant")
                        error[::self.stride_shape[0]] = error_tensor[batch][error_channel]
                    elif self.weights[0][0].ndim == 2:
                        input_tensor = np.pad(self.input_tensor[batch][input_channel],
                                              (self.padding(self.convolution_shape[1]),
                                               self.padding(self.convolution_shape[2])),
                                              mode="constant")
                        error[::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[batch][error_channel]

                    if input_tensor is not None:
                        tensor = scipy.signal.correlate(input_tensor,
                                                       error,
                                                       mode="valid")
                        self.gradient_weights[error_channel][input_channel] += tensor

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return output_tensor

    def rearrange_weights(self):
        weights = np.empty(self.weights.shape)
        weights = np.moveaxis(weights, 0, 1)
        for kernel in range(self.num_kernels):
            for channel in range(self.weights[kernel].shape[0]):
                weights[channel][kernel] = self.weights[kernel][channel]
        return weights

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape),
                                                      np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize(self.bias.shape, np.prod(self.convolution_shape),
                                                np.prod(self.convolution_shape[1:]) * self.num_kernels)
