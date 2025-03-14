import numpy as np

class Linear:

    def __init__(self, input_dim:int, output_dim:int):
        assert output_dim > 0 and input_dim > 0, "inputs or outputs cannot be negative"

        self.input_dim: int = input_dim
        self.neurons: int = output_dim
        self.neuron_w = (
                np.random.randn(self.neurons,self.input_dim) * np.sqrt(2 / self.input_dim)
        )
        self.neuron_b = np.zeros(self.neurons)

        self.input_a: np.ndarray = None
        self.dldw: np.ndarray = None
        self.dldb: np.ndarray = None

    def forward_propagation(self, input_a: np.ndarray) -> np.ndarray:
        assert input_a.shape[-1] == self.input_dim, "input dimensions dont match"
        self.input_a = input_a
        return np.matmul(input_a, self.neuron_w.T) + self.neuron_b

    def backpropagation(self, dldz: np.ndarray) -> np.ndarray:
        assert self.input_a is not None, "dzdw is None"

        dzda = self.neuron_w
        dzdw = self.input_a
        self.dldw = np.matmul(dldz.T, dzdw)
        self.dldb = np.sum(dldz)

        return np.matmul(dldz, dzda) # Sums over all the neurons in the layer ahead of the current layer













