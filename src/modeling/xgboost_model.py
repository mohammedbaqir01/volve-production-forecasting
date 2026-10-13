from xgboost import XGBRegressor


OPTUNA_BEST_PARAMS = {
    "n_estimators": 1425,
    "learning_rate": 0.014817743795218469,
    "max_depth": 7,
    "min_child_weight": 2,
    "subsample": 0.9327640601767555,
    "colsample_bytree": 0.9362119123516623,
    "gamma": 2.998715624279977,
    "reg_alpha": 0.4960576622381044,
    "reg_lambda": 0.0024168957124984596,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_rounds": 50,
}


def build_xgboost_model() -> XGBRegressor:
    """Create the XGBoost estimator with the notebook's best Optuna parameters.

    Returns:
        An unfitted ``XGBRegressor`` configured with the selected Optuna
        parameters and validation-based early stopping.
    """
    return XGBRegressor(**OPTUNA_BEST_PARAMS)


def fit_xgboost(x_train, y_train, x_val, y_val) -> XGBRegressor:
    """Fit the configured estimator on the supplied lag-feature matrix.

    Args:
        x_train: Training data, typically columns ``lag_1`` to
            ``lag_30``.
        y_train: Target values aligned row-for-row with
            ``x_train``.
        x_val: Chronological validation lag features for early stopping.
        y_val: Validation targets aligned row-for-row with ``x_val``.

    Returns:
        The fitted XGBoost regressor.
    """
    model = build_xgboost_model()
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=100)
    return model
