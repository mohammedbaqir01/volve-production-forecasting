# Volve Production Forecasting

A reproducible Python workflow for forecasting daily bore-oil production at the Volve field. It compares ARIMA, XGBoost, and LSTM models using a chronological holdout.

![Daily bore-oil production](figures/02_daily_bore_oil_production.png)

## Quick start

Use Python 3.10–3.12. Install dependencies and run from the repository root:

```bash
python -m pip install -r requirements.txt
python run_pipeline.py
```

The included workbook is `data/Volve production data.xlsx`, sourced from [Equinor's Volve field dataset](https://www.equinor.com/energy/volve-data-sharing). The dataset has terms separate from this repository's code license; follow Equinor's attribution and reuse requirements. The pipeline writes forecast plots to `figures/` and fitted model artifacts to `models/`. The LSTM model uses Keras' native `.keras` format; its scalers and lag metadata are stored separately.

Override paths, split settings, and forecast windows as needed:

```bash
python run_pipeline.py --help
python run_pipeline.py --data-path path/to/workbook.xlsx --test-fraction 0.15 --lags 30 --arima-horizon 30
```

The default train and validation dates match the included Volve dataset. Adjust `--train-end`, `--validation-start`, and `--validation-end` when using data with different coverage.

## Models and evaluation

- **ARIMA (3, 1, 2):** forecasts 30 days from the end of pre-test data; missing dates stay missing, not zero.
- **XGBoost and LSTM:** use 30 previous recorded values. Scalers fit on training data; validation controls early stopping.

XGBoost and LSTM use **rolling one-step-ahead forecasts**: each test prediction uses actual values through the prior test date. Lags count recorded rows; the historical workbook has missing calendar dates, though the test period is daily and continuous.

The chronological split is by recorded rows. Predictor gaps use medians learned through the training cutoff; target values are never imputed, and zero-production days are kept. Scores are not directly comparable: ARIMA uses all pre-test data and the first 30 test days, while XGBoost/LSTM fit through 2014-07-15, validate on later pre-test data, and score the full test period.

## Forecast plots

The notebook produces the following model comparisons:

| Model | Figure |
| --- | --- |
| ARIMA | [Forecast vs. actual](figures/10_arima_forecast_vs_actual.png) |
| XGBoost | [Forecast](figures/12_xgboost_forecast.png) |
| LSTM | [Forecast vs. actual](figures/13_lstm_forecast_vs_actual.png) |

## Repository layout

- `notebooks/time_series.ipynb` — exploration, diagnostics, and model comparison.
- `src/` — reusable loading, feature, modeling, evaluation, and plotting modules.
- `run_pipeline.py` — command-line pipeline entry point.
- `data/` — included Volve source workbook.
- `figures/` — notebook diagnostic and forecast plots.
- `models/` — local model outputs (ignored by Git).

## Development

GitHub Actions runs Ruff lint checks on pushes and pull requests.

For the documented exploratory workflow, open `notebooks/time_series.ipynb`. See [`src/README.md`](src/README.md) for module details.
