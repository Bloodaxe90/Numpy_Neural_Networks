import numpy as np



class ReLU:

    def __init__(self):
        self.dadz: np.ndarray = None

    def forward_propagation(self, input_z: np.ndarray) -> np.ndarray:
        self.dadz = np.where(input_z > 0, 1, 0)
        return np.maximum(0, input_z)

    def backpropagation(self, dldz: np.ndarray) -> np.ndarray:
        assert self.dadz is not None, "dadz is None"
        return dldz * self.dadz



