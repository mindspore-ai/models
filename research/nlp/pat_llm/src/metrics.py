"""Forecasting metrics with stable handling around zero targets."""

import numpy as np


def calculate_metrics(prediction, target, eps=1e-8):
    """Return MAE, MSE, RMSE, MAPE, MSPE, and R2."""
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    error = prediction - target
    mse = np.mean(np.square(error))
    valid = np.abs(target) > eps
    denominator = np.sum(np.square(target - target.mean()))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAPE": float(np.mean(np.abs(error[valid] / target[valid])))
        if valid.any()
        else float("nan"),
        "MSPE": float(np.mean(np.square(error[valid] / target[valid])))
        if valid.any()
        else float("nan"),
        "R2": float(
            1.0 - np.sum(np.square(error)) / max(denominator, eps)
        ),
    }
