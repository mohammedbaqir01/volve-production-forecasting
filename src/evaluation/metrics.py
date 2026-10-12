import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted) -> dict[str, float]:
    """Calculate MAE, RMSE, and R² for aligned actual and predicted values.

    Args:
        actual: Ground-truth target values.
        predicted: Model predictions in the same units and order as ``actual``.

    Returns:
        A dictionary with floating-point ``mae``, ``rmse``, and ``r2`` values.
    """
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }
