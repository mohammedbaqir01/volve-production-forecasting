<p align="center">
  <img src="figures/readme-banner.svg" alt="Volve oil production forecasting for well 15/9-F-12" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10 or later" />
  <img src="https://img.shields.io/badge/Models-ARIMA%20%7C%20XGBoost%20%7C%20LSTM-0f766e" alt="ARIMA, XGBoost, and LSTM models" />
  <img src="https://img.shields.io/badge/License-MIT-22c55e" alt="MIT license" />
</p>

Forecast daily oil production for Volve well `15/9-F-12`. The project uses ARIMA, XGBoost, and LSTM models.

## What’s inside

The notebook cleans and explores the production data, then trains and compares three models:

- **ARIMA:** forecasts 30 days ahead.
- **XGBoost:** uses the previous 30 days to predict production.
- **LSTM:** learns patterns from the previous 30 days and uses early stopping.

## Results

Saved notebook results:

| Model | Test RMSE | Test R² |
| --- | ---: | ---: |
| XGBoost | 52.87 | 0.864 |
| XGBoost, tuned | 50.14 | 0.88 |
| LSTM | 53.70 | 0.860 |

XGBoost and LSTM predict one test day at a time using earlier observed values. ARIMA covers a fixed 30-day period, so its results are not directly comparable. The notebook does not report ARIMA test scores.

## Charts

![Daily oil production](figures/01_daily_bore_oil_production.png)

![Yearly peak oil production](figures/02_annual_maximum_bore_oil_production.png)

![ARIMA forecast](figures/08_arima_forecast_vs_actual.png)

![XGBoost forecast](figures/10_xgboost_forecast_vs_actual.png)

![LSTM forecast](figures/11_lstm_forecast_vs_actual.png)

## Run the notebook

Use Python 3.10 or later and install the packages in `requirements.txt`. Open `notebooks/time_series.ipynb` and run the cells from top to bottom. Update the workbook path if needed. The workbook must have a `Daily Production Data` sheet.

## Project files

- `data/` — production workbook
- `notebooks/` — analysis and model experiments
- `figures/` — charts
- `models/` — saved XGBoost and LSTM models
- `requirements.txt` — Python packages

## License

See [LICENSE](LICENSE).
