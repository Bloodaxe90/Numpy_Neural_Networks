import numpy as np

class Flatten:

    def __init__(self):
        self.input_a: np.ndarray = None

    def forward_propagation(self, input_a: np.ndarray) -> np.ndarray:
        self.input_a = input_a
        return input_a.reshape((self.input_a.shape[0], -1))

    def backpropagation(self, dlda: np.ndarray) -> np.ndarray:
        batch_size, input_channels, height, width = self.input_a.shape

        return dlda.reshape((batch_size, input_channels, height, width))