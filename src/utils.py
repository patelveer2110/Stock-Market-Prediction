# src/utils.py
import numpy as np
import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def make_supervised(df, feature_cols, target_col='Close', lookback=60, horizon=1, scale_method='minmax'):
    """
    Return:
      X_seq (n, lookback, n_features), y (n,1), feat_scaler, target_scaler
    """
    values = df[feature_cols + [target_col]].values
    n_rows = len(df)
    Xs, ys = [], []
    for i in range(lookback, n_rows - horizon + 1):
        Xs.append(values[i-lookback:i, :len(feature_cols)])  # features window
        ys.append(values[i+horizon-1, -1])  # target at horizon
    Xs = np.array(Xs)
    ys = np.array(ys).reshape(-1,1)
    if Xs.size == 0:
        raise ValueError("Not enough rows to create supervised samples. Reduce lookback/horizon or ensure processed data is long enough.")
    # scale
    if scale_method=='minmax':
        feat_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
    else:
        feat_scaler = StandardScaler()
        target_scaler = StandardScaler()
    ns, lb, nf = Xs.shape
    Xs_2d = Xs.reshape(ns, lb*nf)
    Xs_scaled_2d = feat_scaler.fit_transform(Xs_2d)
    Xs_scaled = Xs_scaled_2d.reshape(ns, lb, nf)
    ys_scaled = target_scaler.fit_transform(ys)
    return Xs_scaled, ys_scaled, feat_scaler, target_scaler

def flatten_for_sklearn(X_seq):
    ns, lb, nf = X_seq.shape
    return X_seq.reshape(ns, lb*nf)

def time_train_val_test_split(nsamples, train_frac=0.7, val_frac=0.15):
    ntrain = int(nsamples * train_frac)
    nval = int(nsamples * val_frac)
    nval_end = ntrain + nval
    return ntrain, nval_end

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

def save_sklearn_model(model, path):
    joblib.dump(model, path)

def load_sklearn_model(path):
    return joblib.load(path)

def save_metrics(df_metrics, path):
    ensure_dir(os.path.dirname(path))
    df_metrics.to_csv(path, index=False)

def inverse_transform_targets(preds_scaled, target_scaler):
    import numpy as np
    preds = np.array(preds_scaled).reshape(-1,1)
    return target_scaler.inverse_transform(preds).ravel()
