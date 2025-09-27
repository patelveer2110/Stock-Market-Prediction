import sys
sys.path.append("src")

from data import download_and_save
from features import add_all_indicators

if __name__ == "__main__":
    ticker = "RELIANCE.NS"
    df = download_and_save(ticker, "data/raw")
    df_feat = add_all_indicators(df)
    print(df_feat.head())
