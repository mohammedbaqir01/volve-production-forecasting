"""Train, evaluate, and save the notebook's forecasting models."""

import argparse
from pathlib import Path

from src.data.production_data import load_production_data, prepare_production_data
from src.evaluation.metrics import regression_metrics
from src.features.lag_features import add_lag_features
from src.modeling.arima import fit_arima_forecast
from src.modeling.lstm_model import fit_lstm
from src.modeling.xgboost_model import fit_xgboost
from src.utils.artifact_io import save_artifact
from src.visualization.plots import save_forecast_plot


def run_pipeline(
    data_path: str | Path = "data/Volve production data.xlsx",
    output_dir: str | Path = "models",
    figures_dir: str | Path = "figures",
    test_fraction: float = 0.15,
    train_end: str = "2014-07-15",
    validation_start: str = "2014-08-16",
    validation_end: str = "2015-06-16",
    n_lags: int = 30,
    arima_horizon: int = 30,
):
    """Run the notebook's ARIMA, XGBoost, and LSTM forecasting workflow.

    The workbook is loaded and cleaned, ARIMA forecasts the next 30 days, and
    XGBoost/LSTM are trained on the notebook's chronological 30-lag windows.
    Fitted estimators, LSTM scalers, and forecast figures are saved to the
    requested folders.

    Args:
        data_path: Volve workbook path; defaults to the repository data folder.
        output_dir: Folder for serialized model artifacts.
        figures_dir: Folder for forecast comparison plots.

    Returns:
        A mapping for ``arima``, ``xgboost``, and ``lstm``. Each entry contains
        its forecast or fitted estimator and test metrics; the LSTM entry also
        includes its training history. ARIMA metrics use the forecast dates
        that overlap the available test partition.
    """
    train_df, test_df = load_production_data(data_path, test_fraction=test_fraction)
    train_df, test_df, _ = prepare_production_data(train_df, test_df)

    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ARIMA uses the cleaned training series and predicts the next 30 days.
    arima_predictions = fit_arima_forecast(train_df, steps=arima_horizon)
    arima_actual = test_df["BORE_OIL_VOL"].reindex(arima_predictions.index).dropna()
    arima_predictions = arima_predictions.reindex(arima_actual.index)
    arima_metrics = regression_metrics(arima_actual, arima_predictions)
    save_forecast_plot(
        arima_actual,
        arima_predictions,
        figures_dir / "arima_forecast_vs_actual.png",
        "ARIMA Forecast vs. Actual Bore Oil Production",
    )

    train_df, lag_feature_names = add_lag_features(train_df, n=n_lags)
    test_df, _ = add_lag_features(test_df, n=n_lags)
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    x_train = train_df.loc[:train_end, lag_feature_names]
    y_train = train_df.loc[:train_end, "BORE_OIL_VOL"]
    x_val = train_df.loc[validation_start:validation_end, lag_feature_names]
    y_val = train_df.loc[validation_start:validation_end, "BORE_OIL_VOL"]
    x_test = test_df[lag_feature_names]
    y_test = test_df["BORE_OIL_VOL"]

    xgboost_model = fit_xgboost(x_train, y_train, x_val, y_val)
    xgb_predictions = xgboost_model.predict(x_test)
    xgb_metrics = regression_metrics(y_test, xgb_predictions)
    save_artifact(xgboost_model, output_dir / "xgb_production_forecasting_model.pkl")
    save_forecast_plot(
        y_test,
        xgb_predictions,
        figures_dir / "xgboost_forecast_vs_actual.png",
        "XGBoost Forecast vs. Actual Bore Oil Production",
    )

    lstm_model, history, lstm_predictions, feature_scaler, target_scaler = fit_lstm(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        n_lags=len(lag_feature_names),
    )
    lstm_metrics = regression_metrics(y_test, lstm_predictions)
    save_artifact(
        {"model": lstm_model, "x_scaler": feature_scaler, "y_scaler": target_scaler, "lags": lag_feature_names},
        output_dir / "lstm_production_forecasting_model.pkl",
    )
    save_forecast_plot(
        y_test,
        lstm_predictions,
        figures_dir / "lstm_forecast_vs_actual.png",
        "LSTM Forecast vs. Actual Bore Oil Production",
    )
    return {
        "arima": {"model": arima_predictions, "metrics": arima_metrics},
        "xgboost": {"model": xgboost_model, "metrics": xgb_metrics},
        "lstm": {"model": lstm_model, "history": history, "metrics": lstm_metrics},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast Volve bore-oil production.")
    parser.add_argument("--data-path", default="data/Volve production data.xlsx")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--train-end", default="2014-07-15")
    parser.add_argument("--validation-start", default="2014-08-16")
    parser.add_argument("--validation-end", default="2015-06-16")
    parser.add_argument("--lags", type=int, default=30)
    parser.add_argument("--arima-horizon", type=int, default=30)
    args = parser.parse_args()

    results = run_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        test_fraction=args.test_fraction,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        n_lags=args.lags,
        arima_horizon=args.arima_horizon,
    )
    for model_name, model_result in results.items():
        print(model_name, model_result["metrics"])
