import os
import pickle
import datetime
import pandas as pd


def load_model(model_name: str):
    model_dir = "/Users/Eric/PycharmProjects/NN/models"
    with open(f"{model_dir}/{model_name}.pkl", "rb") as f:
        return pickle.load(f)

def save_model(model_name: str, weights_biases):
    model_dir = "/Users/Eric/PycharmProjects/NN/models"
    with open(f"{model_dir}/{model_name}.pkl", "wb") as f:
        pickle.dump(weights_biases, f)

def save_results(results: pd.DataFrame, extra: str = ""):
    log_dir = "/Users/Eric/PycharmProjects/NN/logs"
    current_datetime = datetime.datetime.now()
    result_dir = f"{log_dir}/{current_datetime.date()}"
    if extra:
        result_dir += f"/{extra}"

    os.makedirs(result_dir, exist_ok= True)

    results.to_csv(f"{result_dir}/{current_datetime.time()}.csv", index= False)

