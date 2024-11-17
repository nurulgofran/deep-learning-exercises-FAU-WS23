import numpy as np

from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, in_tensor):
        self.input_tensor = in_tensor
        return np.maximum(in_tensor, 0)

    def backward(self, error_tensor):
        x = self.input_tensor > 0
        error_tensor *= x
        return error_tensor
