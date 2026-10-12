import pandas as pd


def add_lag_features(
    frame: pd.DataFrame, target: str = "BORE_OIL_VOL", n: int = 30
) -> tuple[pd.DataFrame, list[str]]:
    """Append the previous ``n`` target observations as feature columns.

    Args:
        frame: Time-ordered observations. Callers are responsible for sorting
            by time before building lags.
        target: Name of the series to lag.
        n: Number of historical observations to add.

    Returns:
        A copy of ``frame`` with ``lag_1`` through ``lag_n`` columns, and the
        corresponding ordered list of new column names. The first ``n`` rows
        contain missing lag values and must be excluded or otherwise handled.
    """
    frame = frame.copy()
    lag_feature_names = []
    for i in range(1, n + 1):
        frame[f"lag_{i}"] = frame[target].shift(i)
        lag_feature_names.append(f"lag_{i}")
    return frame, lag_feature_names
