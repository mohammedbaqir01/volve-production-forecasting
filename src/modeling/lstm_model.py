import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler


def fit_lstm(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    n_lags: int,
    epochs: int = 100,
    batch_size: int = 16,
    patience: int = 45,
):
    """Train the notebook's LSTM and produce predictions in target units.

    Scalers are fitted only on the training data. Lag columns are reshaped to
    ``(samples, timesteps, 1)`` and early stopping monitors validation loss.

    Args:
        x_train: Training lag matrix with ``n_lags`` columns.
        y_train: Training target values.
        x_val: Validation lag matrix with the same columns as
            ``x_train``.
        y_val: Validation target values.
        x_test: Test lag matrix with the same columns as
            ``x_train``.
        n_lags: Number of lag values in each sequence.
        epochs: Maximum training epochs.
        batch_size: Number of samples per gradient update.
        patience: Validation epochs without improvement before early stopping.

    Returns:
        A tuple of the fitted model, Keras training history, inverse-scaled
        test predictions, fitted input scaler, and fitted target scaler.
    """
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    x_train_scaled = feature_scaler.fit_transform(x_train)
    x_val_scaled = feature_scaler.transform(x_val)
    x_test_scaled = feature_scaler.transform(x_test)
    y_train_scaled = target_scaler.fit_transform(np.asarray(y_train).reshape(-1, 1))
    y_val_scaled = target_scaler.transform(np.asarray(y_val).reshape(-1, 1))

    x_train_lstm = x_train_scaled.reshape(x_train_scaled.shape[0], n_lags, 1)
    x_val_lstm = x_val_scaled.reshape(x_val_scaled.shape[0], n_lags, 1)
    x_test_lstm = x_test_scaled.reshape(x_test_scaled.shape[0], n_lags, 1)

    model = Sequential()
    model.add(LSTM(64, input_shape=(n_lags, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    history = model.fit(
        x_train_lstm,
        y_train_scaled,
        validation_data=(x_val_lstm, y_val_scaled),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=10,
    )
    predictions_scaled = model.predict(x_test_lstm, verbose=0)
    lstm_predictions = target_scaler.inverse_transform(predictions_scaled).ravel()
    return model, history, lstm_predictions, feature_scaler, target_scaler
