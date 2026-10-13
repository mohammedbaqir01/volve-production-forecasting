import pandas as pd
import statsmodels.api as sm


def fit_arima_forecast(
    train_df: pd.DataFrame,
    target: str = "BORE_OIL_VOL",
    order: tuple[int, int, int] = (3, 1, 2),
    steps: int = 30,
):
    """Fit an ARIMA model and date-index its daily forecasts.

    Args:
        train_df: Time-indexed training observations containing ``target``.
        target: Column to forecast.
        order: ARIMA ``(p, d, q)`` order; defaults to the notebook's setting.
        steps: Number of future daily observations to forecast.

    Returns:
        The fitted statsmodels result object and a forecast series indexed by
        daily timestamps immediately after the final training date.
    """
    training_oil_volume = train_df[target].reset_index(drop=True)
    model = sm.tsa.ARIMA(training_oil_volume, order=order)
    arima_model_fit = model.fit()
    arima_model_predictions = arima_model_fit.forecast(steps=steps)
    arima_model_predictions.index = pd.date_range(
        start=train_df.index[-1] + pd.Timedelta(days=1),
        periods=steps,
        freq="D",
    )
    return arima_model_predictions
