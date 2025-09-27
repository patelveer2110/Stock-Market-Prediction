# src/data.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from src.features import add_all_indicators

DATA_DIR = os.path.join(os.getcwd(), 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def download_ticker(ticker: str, start='2010-01-01', end=None, interval='1d') -> pd.DataFrame:
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")
    df.index = pd.to_datetime(df.index)
    df.to_csv(os.path.join(RAW_DIR, f'{ticker}.csv'))
    return df

def load_raw(ticker: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f'{ticker}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw file not found: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True)

def prepare_and_save(ticker: str, start='2010-01-01', end=None, add_indicators=True, ma_windows=[7,14,21,50,100]):
    # downloads if not present
    raw_path = os.path.join(RAW_DIR, f'{ticker}.csv')
    if not os.path.exists(raw_path):
        download_ticker(ticker, start=start, end=end)
    df = load_raw(ticker)
    df.columns = [c.capitalize() for c in df.columns]
    # keep canonical columns
    df = df[['Open','High','Low','Close','Adj Close','Volume']].rename(columns={'Adj Close':'Adj_Close'})
    if add_indicators:
        df = add_all_indicators(df, ma_windows=ma_windows)
    df = df.dropna()
    processed_path = os.path.join(PROCESSED_DIR, f'{ticker}.csv')
    df.to_csv(processed_path)
    return df

def load_processed(ticker: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f'{ticker}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True)

def download_and_save(ticker: str, out_dir=RAW_DIR, start="2010-01-01", end=None, interval="1d"):
    """
    Simple wrapper: download raw data and save it.
    Returns the dataframe.
    """
    df = download_ticker(ticker, start=start, end=end, interval=interval)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{ticker}.csv")
    df.to_csv(path)
    return df
