# Time-series source modules

The notebook remains the exploratory record. Reusable pieces from its production forecasting workflow are organized here:

- `data/production_data.py` loads the Volve workbook, validates the `15/9-F-12` series, makes the chronological 85/15 split, and imputes numeric predictors using training-fitted medians. Missing targets are rejected.
- `features/lag_features.py` creates lag inputs from previous recorded observations for the tree and recurrent models; the workbook has some missing calendar dates.
- `modeling/arima.py`, `modeling/xgboost_model.py`, and `modeling/lstm_model.py` hold the model setup and fitting code.
- `evaluation/metrics.py` calculates MAE, RMSE, and R².
- `visualization/plots.py` saves the forecast comparison figures.
- `run_pipeline.py` connects data prep, training, evaluation, and artifact/figure saving.
- `utils/artifact_io.py` is the existing model artifact helper.

ARIMA uses a daily index with absent dates left missing. Its metrics cover the first 30 test days; XGBoost and LSTM use rolling one-step predictions over the full test period. Their scores are not directly comparable.

The pipeline saves the LSTM network as a native `.keras` model and stores its fitted scalers and lag names in a separate joblib artifact.

From the repository root, run `python run_pipeline.py`. Use `python run_pipeline.py --help` to customize workbook/output paths, test fraction, lag window, ARIMA horizon, and train/validation date windows. The defaults match the included workbook; update the date windows for other datasets.
