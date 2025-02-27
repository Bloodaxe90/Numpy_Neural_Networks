import numpy as np
from src.nn.activation_functions.relu import ReLU
from src.nn.activation_functions.softmax import Softmax
from src.nn.layers.convolution import Convolution
from src.nn.models.FFNN import FFNN
from src.nn.utils.flatten import Flatten
from src.nn.layers.linear import Linear
from src.nn.utils.max_pool import MaxPool


class CNN:

    def __init__(self, *args, output_dim: int, input_dims: tuple):
        assert len(args) > 1, "Layer dims need more than 1 input"
        self.layers = []
        for i in range(len(args) - 1):
            self.layers.extend([
                Convolution(
                    input_channels= args[i],
                    output_channels= args[i + 1],
                    kernel_size= 3,
                    padding= 1,
                    stride= 1
                ),
                ReLU(),
                MaxPool(
                    kernel_size=2,
                    stride=2
                ),
            ])

        self.layers.append(Flatten())
        conv_layers = len(args) - 1
        flattened_neurons_input = int(((input_dims[0] * input_dims[1]) / (2 **conv_layers)) * 2)

        self.layers.extend(
            FFNN(
                flattened_neurons_input, 100, output_dim
            ).layers
        )

        self.trainable_layers = [
            layer for layer in self.layers
            if isinstance(layer, Linear) or isinstance(layer, Convolution)
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
            if isinstance(layer, Linear):
                layer.neuron_w = weights[idx]
                layer.neuron_b = biases[idx]
            elif isinstance(layer, Convolution):
                layer.kernel_w = weights[idx]
                layer.kernel_b = biases[idx]

    def get_weights_biases(self) -> tuple:
        weights = []
        biases = []
        for layer in self.trainable_layers:
            if isinstance(layer, Linear):
                weights.append(layer.neuron_w)
                biases.append(layer.neuron_b)
            elif isinstance(layer, Convolution):
                weights.append(layer.kernel_w)
                biases.append(layer.kernel_b)
        return weights, biases

