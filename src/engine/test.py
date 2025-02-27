import numpy as np
import pandas as pd
from src.nn.loss_functions.categorical_cross_entropy import \
    categorical_cross_entropy_loss

from src.utils.save_load import save_model, save_results
from src.utils.setup import batch_generator


def test(model,
          test_images: np.ndarray,
          test_labels: np.ndarray,
          epochs: int = 1,
          experiment_name: str = ""
          ) -> pd.DataFrame:

    print("Testing Began")
    results = pd.DataFrame(columns = ["epoch", "loss", "accuracy"])

    for epoch in range(epochs):
        total_accuracy = 0
        total_loss = 0
        for idx, image in enumerate(test_images):
            y_prob = model.forward_pass(np.expand_dims(image, axis=0))

            y_pred = y_prob
            y_actual = np.expand_dims(test_labels[idx], axis=0)
            total_accuracy += np.sum(
                np.argmax(y_pred, axis= 1) == np.argmax(y_actual, axis= 1)
            )
            loss, dldy = categorical_cross_entropy_loss(y_pred, y_actual)
            total_loss += sum(loss)

            if idx % 1000 == 0 and idx != 0:
                avg_loss = total_loss / idx
                avg_accuracy = total_accuracy / idx

                log_message: str = f"Image {idx} | "\
                                   f"Loss {avg_loss} | "\
                                   f"Accuracy {avg_accuracy}"
                print(log_message)
                results.loc[len(results)] = [epoch, avg_loss,
                                             avg_accuracy]

    save_results(results, extra= experiment_name)
    return results




