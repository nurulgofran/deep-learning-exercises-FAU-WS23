import copy


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        tensor = self.data_layer.next()
        input_tensor = tensor[0]
        self.label_tensor = tensor[1]

        output_tensor = input_tensor
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)

        return self.loss_layer.forward(output_tensor, self.label_tensor)

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)

        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return input_tensor
