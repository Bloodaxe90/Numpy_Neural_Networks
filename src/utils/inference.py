import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.nn import models


def view_MNIST_image(images: np.ndarray, labels: np.ndarray, idx: int = -1):
    if idx == -1:
        idx = random.randint(0,len(labels))

    plt.imshow(images[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {np.argmax(labels[idx])}")
    plt.show()

def plot_test_results(test_results: pd.DataFrame):

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(test_results.index, test_results["loss"], label="Train Loss")
    ax[0].set_title("Test Loss")
    ax[0].set_xlabel("Batch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(test_results.index, test_results["accuracy"], label="Train Accuracy", color='green')
    ax[1].set_title("Test Accuracy")
    ax[1].set_xlabel("Image")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    fig.suptitle("Training and Testing Metrics Over Epochs", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def show_predictions(model: models, test_images: np.ndarray, test_labels: np.ndarray, num_predictions: int = 9):
    plt.figure(figsize=(10, 10))
    for i in range(num_predictions):
        idx = random.randint(0,len(test_labels))
        image = test_images[idx]
        y_prob = model.forward_pass(np.expand_dims(image, axis=0))

        y_pred = y_prob
        y_actual = np.expand_dims(test_labels[idx], axis=0)

        plt.subplot(3, 3, i + 1)
        plt.title(f"Prediction: {np.argmax(y_pred)}\nActual Label: {np.argmax(y_actual)}")
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.axis("off")
    plt.show()