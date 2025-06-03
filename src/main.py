import os

from src.nn import models, optimizers
from src.nn.loss_functions.categorical_cross_entropy import \
    categorical_cross_entropy_loss
from src.nn.models.CNN import CNN
from src.nn.models.FFNN import FFNN
from src.nn.optimizers.mini_batch_gradient_descent import \
    MiniBatchGradientDescent
from src.nn.optimizers.stochastic_gradient_descent import \
    StochasticGradientDescent
from src.engine.train import train
from src.utils.setup import load_MNIST


def main():
    DATASET: str = "Fashion_MNIST"
    ONE_DIM: bool = False

    # DATASET: str = "MNIST"
    # ONE_DIM: bool = True


    root_path = f"{os.path.dirname(os.getcwd())}/resources/{DATASET}"
    train_images, train_labels, test_images, test_labels = load_MNIST(
        root_path, one_dim=ONE_DIM)
    classes = train_labels.shape[-1]


    MODEL: models = CNN(
        1, 4, 8, output_dim=classes,
        input_dims=tuple(test_images.shape[-2:])
    )
    LEARNING_RATE: float = 0.0001
    OPTIMIZER: optimizers = StochasticGradientDescent(lr=LEARNING_RATE)
    EPOCHS: int = 0
    BATCH_SIZE: int = 1
    MODEL_NAME: str = "tset"

    # MODEL: models = FFNN(
    #         len(train_images[-1]), 700, 500, 700, classes
    #     )
    # LEARNING_RATE: float = 0.0001
    # OPTIMIZER: optimizers = StochasticGradientDescent(lr= LEARNING_RATE)
    # EPOCHS: int = 4
    # BATCH_SIZE: int = 1
    # MODEL_NAME: str = "FFNN_MNIST_SGD_Batch1_Epoch4_lr.0001"

    train(model=MODEL,
          train_images=train_images,
          train_labels=train_labels,
          optimizer=OPTIMIZER,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          model_name=MODEL_NAME
          )

if __name__ == "__main__":
    main()