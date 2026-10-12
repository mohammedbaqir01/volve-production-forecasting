from xgboost import XGBRegressor


def build_xgboost_model() -> XGBRegressor:
    """Create the notebook-configured XGBoost regression estimator.

    Returns:
        An unfitted ``XGBRegressor`` using squared-error loss, 500 trees,
        learning rate 0.05, maximum depth 4, and random seed 42.
    """
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )


def fit_xgboost(x_train, y_train) -> XGBRegressor:
    """Fit the configured estimator on the supplied lag-feature matrix.

    Args:
        x_train: Training data, typically columns ``lag_1`` to
            ``lag_30``.
        y_train: Target values aligned row-for-row with
            ``x_train``.

    Returns:
        The fitted XGBoost regressor.
    """
    model = build_xgboost_model()
    model.fit(x_train, y_train, verbose=100)
    return model
