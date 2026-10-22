import pandas as pd
import statsmodels.api as sm


def fit_arima_forecast(
    train_df: pd.DataFrame,
    target: str = "BORE_OIL_VOL",
    order: tuple[int, int, int] = (3, 1, 2),
    steps: int = 30,
    maxiter: int = 200,
):
    """Fit a daily ARIMA model and preserve calendar gaps as missing observations.

    Args:
        train_df: Training observations indexed by unique, increasing dates and
            containing ``target``. Missing calendar days are represented as
            missing observations rather than as zero production.
        target: Column to forecast.
        order: ARIMA ``(p, d, q)`` order; defaults to the notebook's setting.
        steps: Number of future daily observations to forecast.
        maxiter: Maximum likelihood optimizer iteration limit.

    Returns:
        A forecast series indexed by daily timestamps immediately after the
        final training date. The state-space likelihood handles missing
        calendar-day observations in the training series.
    """
    if not isinstance(train_df.index, pd.DatetimeIndex):
        raise TypeError("train_df must have a DatetimeIndex for daily ARIMA.")
    if not train_df.index.is_monotonic_increasing or not train_df.index.is_unique:
        raise ValueError("ARIMA training dates must be unique and increasing.")
    if steps < 1:
        raise ValueError("steps must be a positive integer.")
    if maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")

    training_oil_volume = train_df[target].asfreq("D")
    model = sm.tsa.ARIMA(training_oil_volume, order=order)
    arima_model_fit = model.fit(method_kwargs={"maxiter": maxiter})
    arima_model_predictions = arima_model_fit.forecast(steps=steps)
    return arima_model_predictions
