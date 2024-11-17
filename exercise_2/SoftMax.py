import numpy as np

from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.sm_out = None

    def forward(self, in_tensor):
        self.input_tensor = in_tensor
        updated_tensor = in_tensor - np.max(in_tensor, axis=1, keepdims=True)
        exp = np.exp(updated_tensor)
        self.sm_out = exp / np.sum(exp, axis=1, keepdims=True)
        return self.sm_out

    def backward(self, error_tensor):
        tensor = self.sm_out * (error_tensor - np.sum(error_tensor * self.sm_out, axis=1, keepdims=True))
        return tensor
