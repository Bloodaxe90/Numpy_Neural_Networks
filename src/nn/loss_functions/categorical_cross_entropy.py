import numpy as np

def categorical_cross_entropy_loss(y_pred: np.ndarray, y_actual: np.ndarray) -> tuple:
    dldz = y_pred - y_actual #If softmax used to get y_pred, skipped dlda / dadz as dadz is complex
    epsilon = 1e-9  # prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - np.sum(y_actual * np.log(y_pred), axis= 1), dldz
