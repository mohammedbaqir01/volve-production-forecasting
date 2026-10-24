# Source modules

Run the end-to-end workflow from the repository root:

```bash
python run_pipeline.py
```

The workbook defaults to `data/Volve production data.xlsx`; models and plots are saved in `models/` and `figures/`. Custom paths can be passed to `run_pipeline`.

Modules are grouped by purpose: `data/` loads and prepares production records, `features/` builds lag inputs, `modeling/` fits ARIMA/XGBoost/LSTM, `evaluation/` computes metrics, `visualization/` creates plots, and `utils/` handles artifacts.
