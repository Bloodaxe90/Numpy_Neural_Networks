import numpy as np

from src.nn.layers.convolution import Convolution
from src.nn.layers.linear import Linear


class MiniBatchGradientDescent:

    def __init__(self, lr: float = 0.0001):
        self.lr = lr

    def optimize(self, layer: Linear):
        if isinstance(layer, Linear):
            batch_size = layer.neuron_w.shape[0]
            layer.neuron_w -= self.lr * (layer.dldw / batch_size)
            layer.neuron_b -= self.lr * (layer.dldb / batch_size)
        elif isinstance(layer, Convolution):
            batch_size = layer.kernel_w.shape[0]
            layer.kernel_w -= self.lr * (layer.dldw / batch_size)
            layer.kernel_b -= self.lr * (layer.dldb / batch_size)