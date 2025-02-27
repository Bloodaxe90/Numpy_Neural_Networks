import numpy as np



class Softmax:

    def forward_propagation(self, input_z: np.ndarray) -> np.ndarray:
        e_x = np.exp(input_z - np.max(input_z, axis=-1, keepdims=True))
        output_a = e_x / np.sum(e_x, axis=-1, keepdims=True)

        return output_a

    def backpropagation(self, dldz: np.ndarray) -> np.ndarray:
        return dldz
