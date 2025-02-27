import random

import matplotlib.pyplot as plt
import numpy as np


def view_MNIST_image(images: np.ndarray, labels: np.ndarray, idx: int = -1):
    if idx == -1:
        idx = random.randint(0,len(labels))

    plt.imshow(images[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {np.argmax(labels[idx])}")
    plt.show()

