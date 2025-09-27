# src/train.py
import argparse
import os
import pandas as pd
import numpy as np
from .data import prepare_and_save, load_processed, PROCESSED_DIR
from .features import add_all_indicators
from .utils import make_supervised, flatten_for_sklearn, time_train_val_test_split, ensure_dir, save_scaler, save_sklearn_model, save_metrics, inverse_transform_targets
from .models_sklearn import build_svr, build_rf, build_xgb
from .models_keras import build_lstm, build_lstm_cnn, fit_keras_model
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

MODELS_DIR = os.path.join(os.getcwd(), 'models')
RESULTS_DIR = os.path.join(os.getcwd(), 'results')

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def train_for_ticker(ticker,
                     models=['svm','rf','xgb','lstm','lstm_cnn'],
                     use_indicators=True,
                     lookback=60,
                     short_horizon=1,
                     long_horizon=30,
                     train_frac=0.7,
                     val_frac=0.15,
                     scale_method='minmax',
                     epochs=50,
                     batch_size=64,
                     save_models=True,
                     n_jobs=1):
    print(f"[INFO] Preparing data for {ticker}")
    df = load_processed(ticker)
    if use_indicators:
        # processed already includes indicators if prepared by prepare_and_save
        pass
    # select features
    feature_cols = [c for c in df.columns if c != 'Close']
    results = []
    for horizon in [short_horizon, long_horizon]:
        print(f"[INFO] Building supervised data: lookback={lookback}, horizon={horizon}")
        X_seq, y_seq, feat_scaler, target_scaler = make_supervised(df, feature_cols, target_col='Close',
                                                                   lookback=lookback, horizon=horizon, scale_method=scale_method)
        nsamples = X_seq.shape[0]
        ntrain, nval_end = time_train_val_test_split(nsamples, train_frac=train_frac, val_frac=val_frac)
        X_train_seq = X_seq[:ntrain]; y_train = y_seq[:ntrain]
        X_val_seq = X_seq[ntrain:nval_end]; y_val = y_seq[ntrain:nval_end]
        X_test_seq = X_seq[nval_end:]; y_test = y_seq[nval_end:]
        # flatten for sklearn
        X_train = flatten_for_sklearn(X_train_seq)
        X_val = flatten_for_sklearn(X_val_seq)
        X_test = flatten_for_sklearn(X_test_seq)

        # save scalers
        scaler_dir = os.path.join(MODELS_DIR, ticker)
        ensure_dir(scaler_dir)
        save_scaler(feat_scaler, os.path.join(scaler_dir, f'feat_scaler_h{horizon}.joblib'))
        save_scaler(target_scaler, os.path.join(scaler_dir, f'target_scaler_h{horizon}.joblib'))

        # SKLEARN MODELS
        if 'svm' in models:
            print("[INFO] Training SVR")
            svr = build_svr()
            svr.fit(X_train, y_train.ravel())
            preds_s = svr.predict(X_test)
            preds_s_inv = inverse_transform_targets(preds_s, target_scaler)
            y_true_inv = inverse_transform_targets(y_test, target_scaler)
            r = {'ticker':ticker,'model':'svm','horizon':horizon,'RMSE':rmse(y_true_inv,preds_s_inv),'MAE':mae(y_true_inv,preds_s_inv)}
            results.append(r)
            if save_models:
                save_sklearn_model(svr, os.path.join(scaler_dir, f'{ticker}_svm_h{horizon}.joblib'))

        if 'rf' in models:
            print("[INFO] Training RandomForest")
            rf = build_rf(n_estimators=200, n_jobs=n_jobs)
            rf.fit(X_train, y_train.ravel())
            preds_rf = rf.predict(X_test)
            preds_rf_inv = inverse_transform_targets(preds_rf, target_scaler)
            y_true_inv = inverse_transform_targets(y_test, target_scaler)
            r = {'ticker':ticker,'model':'rf','horizon':horizon,'RMSE':rmse(y_true_inv,preds_rf_inv),'MAE':mae(y_true_inv,preds_rf_inv)}
            results.append(r)
            if save_models:
                save_sklearn_model(rf, os.path.join(scaler_dir, f'{ticker}_rf_h{horizon}.joblib'))

        if 'xgb' in models:
            print("[INFO] Training XGBoost")
            xgb = build_xgb(n_estimators=300)
            xgb.fit(X_train, y_train.ravel(), eval_set=[(X_val, y_val.ravel())], early_stopping_rounds=20, verbose=False)
            preds_x = xgb.predict(X_test)
            preds_x_inv = inverse_transform_targets(preds_x, target_scaler)
            y_true_inv = inverse_transform_targets(y_test, target_scaler)
            r = {'ticker':ticker,'model':'xgb','horizon':horizon,'RMSE':rmse(y_true_inv,preds_x_inv),'MAE':mae(y_true_inv,preds_x_inv)}
            results.append(r)
            if save_models:
                save_sklearn_model(xgb, os.path.join(scaler_dir, f'{ticker}_xgb_h{horizon}.joblib'))

        # KERAS MODELS (use sequence inputs)
        n_features = X_seq.shape[2]
        if 'lstm' in models:
            print("[INFO] Training LSTM (this may be slow on CPU)")
            model = build_lstm(lookback, n_features, units=64, dropout=0.2)
            save_path = os.path.join(MODELS_DIR, ticker, f'{ticker}_lstm_h{horizon}.h5')
            model, history = fit_keras_model(model, X_train_seq, y_train, X_val_seq, y_val,
                                             save_path=save_path, epochs=epochs, batch_size=batch_size, patience=10, verbose=0)
            preds = model.predict(X_test_seq).ravel()
            preds_inv = inverse_transform_targets(preds, target_scaler)
            y_true_inv = inverse_transform_targets(y_test, target_scaler)
            r = {'ticker':ticker,'model':'lstm','horizon':horizon,'RMSE':rmse(y_true_inv,preds_inv),'MAE':mae(y_true_inv,preds_inv)}
            results.append(r)

        if 'lstm_cnn' in models:
            print("[INFO] Training LSTM-CNN (this may be slow on CPU)")
            model = build_lstm_cnn(lookback, n_features, filters=64, kernel_size=3, lstm_units=64)
            save_path = os.path.join(MODELS_DIR, ticker, f'{ticker}_lstmcnn_h{horizon}.h5')
            model, history = fit_keras_model(model, X_train_seq, y_train, X_val_seq, y_val,
                                             save_path=save_path, epochs=epochs, batch_size=batch_size, patience=10, verbose=0)
            preds = model.predict(X_test_seq).ravel()
            preds_inv = inverse_transform_targets(preds, target_scaler)
            y_true_inv = inverse_transform_targets(y_test, target_scaler)
            r = {'ticker':ticker,'model':'lstm_cnn','horizon':horizon,'RMSE':rmse(y_true_inv,preds_inv),'MAE':mae(y_true_inv,preds_inv)}
            results.append(r)

    # Save results table
    df_res = pd.DataFrame(results)
    ensure_dir(RESULTS_DIR)
    res_path = os.path.join(RESULTS_DIR, f'{ticker}_results.csv')
    df_res.to_csv(res_path, index=False)
    print(f"[INFO] Results saved to {res_path}")
    return df_res

def prepare_ticker_cli(ticker, start='2010-01-01', add_indicators=True):
    print(f"[INFO] Preparing and saving processed data for {ticker}")
    df = prepare_and_save(ticker, start=start, add_indicators=add_indicators)
    print(f"[INFO] Saved processed to data/processed/{ticker}.csv")
    return df

def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Train models for stock prediction")
    parser.add_argument('--mode', choices=['prepare','train'], required=True)
    parser.add_argument('--ticker', type=str, required=True)
    parser.add_argument('--start', type=str, default='2010-01-01')
    parser.add_argument('--models', type=str, default='svm,rf,xgb,lstm,lstm_cnn')
    parser.add_argument('--lookback', type=int, default=60)
    parser.add_argument('--short_horizon', type=int, default=1)
    parser.add_argument('--long_horizon', type=int, default=30)
    parser.add_argument('--use_indicators', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_frac', type=float, default=0.7)
    parser.add_argument('--val_frac', type=float, default=0.15)
    parser.add_argument('--scale', type=str, default='minmax')
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare_ticker_cli(args.ticker, start=args.start, add_indicators=args.use_indicators)
    elif args.mode == 'train':
        models_list = args.models.split(',')
        df_res = train_for_ticker(args.ticker,
                                  models=models_list,
                                  use_indicators=args.use_indicators,
                                  lookback=args.lookback,
                                  short_horizon=args.short_horizon,
                                  long_horizon=args.long_horizon,
                                  train_frac=args.train_frac,
                                  val_frac=args.val_frac,
                                  scale_method=args.scale,
                                  epochs=args.epochs,
                                  batch_size=args.batch_size,
                                  n_jobs=args.n_jobs)
        print(df_res)

if __name__ == '__main__':
    parse_args_and_run()
