# utils/forecast.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import os

# Prophet import wrapped to provide clear error if not installed
try:
    from prophet import Prophet
except Exception as e:
    Prophet = None

# -------------------------
# Utility: prepare series
# -------------------------
def prepare_series(df, target):
    """
    Input: df must have a datetime column named 'timestamp' (datetime64) or index as datetime.
    Returns: dataframe with columns ['ds','y'] for Prophet (no NaNs)
    """
    if 'timestamp' in df.columns:
        tmp = df[['timestamp', target]].dropna().copy()
        tmp = tmp.rename(columns={'timestamp': 'ds', target: 'y'})
    else:
        # assume index is datetime
        tmp = df[[target]].dropna().reset_index().rename(columns={'index': 'ds', target: 'y'})
    tmp = tmp.sort_values('ds').reset_index(drop=True)
    return tmp

# -------------------------
# Prophet workflow
# -------------------------
def run_prophet_forecast(df, target, train_frac=0.8, periods=30, freq='D', yearly=True, weekly=True, daily=False):
    """
    Runs Prophet on `target` column of df.
    Returns: model, train_df, test_df, forecast_df, metrics dict
    """
    if Prophet is None:
        raise ImportError("Prophet is not installed. Install via: pip install prophet")

    series = prepare_series(df, target)
    if series.empty:
        raise ValueError("No data available for target: " + target)

    # train/test split by time
    n = len(series)
    if n < 10:
        raise ValueError("Too few rows for forecasting. Need more data points.")
    split_idx = int(n * train_frac)
    train_df = series.iloc[:split_idx].copy()
    test_df = series.iloc[split_idx:].copy()

    # init & fit
    m = Prophet(daily_seasonality=daily, weekly_seasonality=weekly, yearly_seasonality=yearly)
    m.fit(train_df)

    # create future frame covering test + horizon
    future_periods = len(test_df) + periods
    future = m.make_future_dataframe(periods=future_periods, freq=freq)
    forecast = m.predict(future)

    # evaluation on test range
    # align forecast yhat with test_df ds values
    forecast_idxed = forecast.set_index('ds')
    test_ds = test_df['ds']
    # find rows that overlap
    available = forecast_idxed.loc[test_ds]
    y_true = test_df['y'].values
    y_pred = available['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    metrics = {'mae': float(mae), 'rmse': float(rmse), 'train_size': len(train_df), 'test_size': len(test_df)}

    return m, train_df, test_df, forecast, metrics

def save_prophet_model(model, path="models/prophet_model.pkl"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    return path

def load_prophet_model(path="models/prophet_model.pkl"):
    return joblib.load(path)


# -------------------------
# Optional: simple LSTM helper
# -------------------------
def create_sequences(data, seq_len=30):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

def train_lstm_pm(df, target='pm25', seq_len=30, epochs=50, batch_size=32, validation_split=0.1):
    """
    Simple LSTM training function (requires tensorflow).
    Returns model, scaler_path, stats
    """
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    except Exception:
        raise ImportError("TensorFlow or sklearn not installed. Install tensorflow and scikit-learn to use LSTM option.")

    series = df[[target]].dropna()
    arr = series.values.astype('float32')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)

    X, y = create_sequences(scaled, seq_len)
    if len(X) < 10:
        raise ValueError("Not enough sequences for LSTM. Increase data or reduce seq_len.")

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    os.makedirs("models", exist_ok=True)
    ckpt = ModelCheckpoint("models/lstm_best.h5", save_best_only=True, monitor='val_loss')
    es = EarlyStopping(patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[ckpt, es])

    # save scaler
    joblib.dump(scaler, "models/pm_scaler.save")
    return model, "models/pm_scaler.save"
