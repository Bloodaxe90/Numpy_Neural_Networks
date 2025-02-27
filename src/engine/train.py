import numpy as np
import pandas as pd
from src.nn.loss_functions.categorical_cross_entropy import \
    categorical_cross_entropy_loss
from src.nn.models.FFNN import FFNN
from src.nn.optimizers.stochastic_gradient_descent import StochasticGradientDescent
from src.utils.save_load import save_model, save_results
from src.utils.setup import batch_generator


def train(model,
          optimizer,
          train_images: np.ndarray,
          train_labels: np.ndarray,
          epochs: int = 1,
          batch_size: int = 32,
          model_name: str = ""
          ):

    print("Training Began")
    results = pd.DataFrame(columns = ["epoch", "batch", "loss", "accuracy"])

    for epoch in range(epochs):
        total_accuracy = 0
        total_loss = 0
        for batch_num, batch_data in enumerate(batch_generator(
                train_images,
                train_labels,
                batch_size
        )):
            train_images_batch, train_labels_batch = batch_data
            y_prob = model.forward_pass(train_images_batch)

            y_pred = y_prob
            y_actual = train_labels_batch
            total_accuracy += np.sum(
                np.argmax(y_pred, axis= 1) == np.argmax(y_actual, axis= 1)
            )
            loss, dldy = categorical_cross_entropy_loss(y_pred, y_actual)
            total_loss += sum(loss)
            model.backward_pass(dldy)
            model.optimization(optimizer= optimizer)

            if (sample_num := (batch_size * batch_num)) % 128 == 0 and sample_num != 0:
                avg_loss = total_loss / sample_num
                avg_accuracy = total_accuracy / sample_num

                log_message: str = f"Batch {batch_num}/{int(len(train_images)/batch_size)} | "\
                                   f"Loss {avg_loss} | "\
                                   f"Accuracy {avg_accuracy}"
                print(log_message)
                results.loc[len(results)] = [epoch, batch_num, avg_loss,
                                             avg_accuracy]

    if model_name:
        save_model(model_name, model.get_weights_biases())

    print(results)
    save_results(results)




