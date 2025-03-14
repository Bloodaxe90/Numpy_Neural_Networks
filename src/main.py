from src.nn.models.CNN import CNN
from src.nn.models.FFNN import FFNN
from src.nn.optimizers.mini_batch_gradient_descent import \
    MiniBatchGradientDescent
from src.nn.optimizers.stochastic_gradient_descent import \
    StochasticGradientDescent
from src.engine.train import train
from src.utils.setup import load_MNIST


def main(cnn: bool = True):
    if cnn:
        root_path = "/Users/Eric/PycharmProjects/NN/resources/Fashion_MNIST"
        train_images, train_labels, test_images, test_labels = load_MNIST(root_path, one_dim= False)
        classes = train_labels.shape[-1]

        optimizer = MiniBatchGradientDescent(lr= 0.001)

        model = CNN(
            1, 4, 8, output_dim= classes, input_dims = tuple(test_images.shape[-2:])
        )

        train(model= model,
              optimizer= optimizer,
              train_images= train_images,
              train_labels= train_labels,
              epochs= 2,
              batch_size= 32,
              model_name= "CNN_Fashion_MBGD_Channel48_Batch32_Epoch2_lr.001"
        )
    else:
        root_path = "/Users/Eric/PycharmProjects/NN/resources/MNIST"
        train_images, train_labels, test_images, test_labels = load_MNIST(root_path)
        classes = train_labels.shape[-1]

        optimizer = StochasticGradientDescent(lr= 0.0001)

        model = FFNN(
            len(train_images[-1]), 700, 500, 700, classes
        )


        train(model = model,
             optimizer = optimizer,
             train_images= train_images,
             train_labels= train_labels,
             epochs=4,
             batch_size=1,
             model_name= "FFNN_MNIST_SGD_Batch1_Epoch4_lr.0001"
        )

if __name__ == "__main__":
    main(cnn = False)