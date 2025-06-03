import os
import pickle
import datetime
import pandas as pd


def load_model(model_name: str):
    model_dir = f"{os.path.dirname(os.getcwd())}/models"
    with open(f"{model_dir}/{model_name}.pkl", "rb") as f:
        return pickle.load(f)

def save_model(model_name: str, weights_biases):
    model_dir = f"{os.path.dirname(os.getcwd())}/models"
    with open(f"{model_dir}/{model_name}.pkl", "wb") as f:
        pickle.dump(weights_biases, f)

def save_results(results: pd.DataFrame, extra: str = ""):
    log_dir = f"{os.path.dirname(os.getcwd())}/logs"
    current_datetime = datetime.datetime.now()
    result_dir = f"{log_dir}/{current_datetime.date()}"
    if extra:
        result_dir += f"/{extra}"

    os.makedirs(result_dir, exist_ok= True)

    results.to_csv(f"{result_dir}/{current_datetime.time()}.csv", index= False)

def load_results(model_name: str) -> pd.DataFrame:
    log_dir = f"{os.path.dirname(os.getcwd())}/logs"

    os.makedirs(log_dir, exist_ok=True)

    if ".csv" not in model_name:
        model_name += ".csv"
    print(f"Loaded {model_name} from directory: {log_dir}")
    return pd.read_csv(f"{log_dir}/{model_name}")

