import idx2numpy
import numpy as np


def load_MNIST(root_path: str, one_dim: bool = True) -> tuple:
    try:
        train_images = idx2numpy.convert_from_file(f"{root_path}/train-images"
                                                   f"-idx3-ubyte")
        train_labels = idx2numpy.convert_from_file(f"{root_path}/train-labels-idx1-ubyte")

        test_images = idx2numpy.convert_from_file(f"{root_path}/test-images-idx3-ubyte")
        test_labels = idx2numpy.convert_from_file(f"{root_path}/test-labels-idx1-ubyte")
    except IOError:
        raise IOError(f"Failed to load a file")

    train_images = (train_images.astype('float32') / 255.0)
    test_images = (test_images.astype('float32') / 255.0)

    if one_dim:
        train_images = train_images.reshape(-1, 28 * 28)
        test_images = test_images.reshape(-1, 28 * 28)
    else:
        train_images = np.expand_dims(train_images, axis= 1)
        test_images = np.expand_dims(test_images, axis= 1)

    one_hot_train_labels = np.zeros((train_labels.shape[0], 10))
    one_hot_test_labels = np.zeros((test_labels.shape[0], 10))
    one_hot_train_labels[np.arange(train_labels.shape[0]), train_labels] = 1
    one_hot_test_labels[np.arange(test_labels.shape[0]), test_labels] = 1
    train_labels = one_hot_train_labels
    test_labels = one_hot_test_labels


    return train_images, train_labels, test_images, test_labels


def batch_generator(images: np.ndarray, labels: np.ndarray,
                    batch_size: int):
    num_samples = len(images)

    for i in range(0, num_samples, batch_size):
        yield images[i:i + batch_size], labels[i:i + batch_size]




