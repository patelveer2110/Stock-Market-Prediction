# src/features.py
import pandas as pd
import numpy as np

def add_moving_averages(df: pd.DataFrame, windows=[7,14,21,50,100]) -> pd.DataFrame:
    for w in windows:
        df[f'SMA_{w}'] = df['Close'].rolling(window=w).mean()
        df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
    return df

def add_stochastic(df: pd.DataFrame, k_window=14, d_window=3) -> pd.DataFrame:
    low_min = df['Low'].rolling(k_window).min()
    high_max = df['High'].rolling(k_window).max()
    df['stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(d_window).mean()
    return df

def add_trix(df: pd.DataFrame, n=15) -> pd.DataFrame:
    ema1 = df['Close'].ewm(span=n, adjust=False).mean()
    ema2 = ema1.ewm(span=n, adjust=False).mean()
    ema3 = ema2.ewm(span=n, adjust=False).mean()
    df['trix'] = ema3.pct_change() * 100
    return df

def add_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    df['return_1d'] = df['Close'].pct_change()
    df['vol_change'] = df['Volume'].pct_change()
    return df

def add_all_indicators(df: pd.DataFrame,
                       add_stoch=True,
                       add_trx=True,
                       ma_windows=[7,14,21,50,100]) -> pd.DataFrame:
    df = df.copy()
    df = add_moving_averages(df, ma_windows)
    if add_stoch:
        df = add_stochastic(df)
    if add_trx:
        df = add_trix(df)
    df = add_basic_returns(df)
    df = df.dropna()
    return df
