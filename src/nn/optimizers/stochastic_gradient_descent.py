from src.nn.layers.convolution import Convolution
from src.nn.layers.linear import Linear

class StochasticGradientDescent:

    def __init__(self, lr: float = 0.0001):
        self.lr = lr

    def optimize(self, layer):
        if isinstance(layer, Linear):
            layer.neuron_w -= self.lr * layer.dldw
            layer.neuron_b -= self.lr * layer.dldb
        elif isinstance(layer, Convolution):
            layer.kernel_w -= self.lr * layer.dldw
            layer.kernel_b -= self.lr * layer.dldb

