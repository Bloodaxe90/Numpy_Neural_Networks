import numpy as np
from src.nn.activation_functions.relu import ReLU
from src.nn.activation_functions.softmax import Softmax
from src.nn.layers.linear import Linear


class FFNN:

    def __init__(self, *args):
        assert len(args) > 1, "Layer dims need more than 1 input"

        self.layers = []
        for i in range(len(args) - 1):
            self.layers.append(Linear(args[i], args[i + 1]))
            if i + 1 != len(args) - 1:
                self.layers.append(ReLU())
        self.layers.append(Softmax())
        self.trainable_layers = [
            layer for layer in self.layers
            if isinstance(layer, Linear)
        ]

    def forward_pass(self, input: np.ndarray) -> np.ndarray:
        x = input
        for layer in self.layers:
            x = layer.forward_propagation(x)
        return x

    def backward_pass(self, dldy) -> np.ndarray:
        g = dldy
        for layer in reversed(self.layers):
            g = layer.backpropagation(g)
        return g

    def optimization(self, optimizer):
        for layer in self.trainable_layers:
            optimizer.optimize(layer)

    def set_weights_biases(self, weight_biases: tuple):
        weights, biases = weight_biases
        for idx, layer in enumerate(self.trainable_layers):
            layer.neuron_w = weights[idx]
            layer.neuron_b = biases[idx]

    def get_weights_biases(self) -> tuple:
        weights = [layer.neuron_w for layer in self.trainable_layers]
        biases = [layer.neuron_b for layer in self.trainable_layers]
        return weights, biases

