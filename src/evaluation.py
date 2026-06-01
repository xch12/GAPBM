##MAE等指标

import numpy as np

def calculate_mae(x1, y1, x2, y2):

    error = np.mean(
        np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    )

    return error

def calculate_rmse(x1, y1, x2, y2):

    rmse = np.sqrt(
        np.mean((x1 - x2)**2 + (y1 - y2)**2)
    )

    return rmse