# Time-series source modules

The notebook remains the exploratory record. Reusable pieces from its production forecasting workflow are organized here:

- `data/production_data.py` loads the Volve workbook, selects the production-only `15/9-F-12` well, makes the chronological 85/15 split, and applies training-fitted median imputation.
- `features/lag_features.py` creates the 30-day lag inputs used by the tree and recurrent models.
- `modeling/arima.py`, `modeling/xgboost_model.py`, and `modeling/lstm_model.py` hold the model setup and fitting code.
- `evaluation/metrics.py` calculates MAE, RMSE, and R².
- `visualization/plots.py` saves the forecast comparison figures.
- `run_pipeline.py` connects data prep, training, evaluation, and artifact/figure saving.
- `utils/artifact_io.py` is the existing model artifact helper.

From the repository root, run `python -m src.run_pipeline`. The workbook defaults to `data/Volve production data.xlsx`; saved models go to `models/`, and forecast plots go to `figures/`.
