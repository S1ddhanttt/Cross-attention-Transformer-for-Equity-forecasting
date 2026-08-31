import os
import yfinance as yf
import pandas as pd
import numpy as np

# Configuration
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
OUTPUT_DIR = "data/processed_market"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes stationary technical features and target forward returns."""
    df = df.copy()
    
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    
    df["normalized_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["normalized_body"] = (df["Close"] - df["Open"]) / df["Open"]
    
    rolling_vol_mean = df["Volume"].rolling(window=20).mean()
    rolling_vol_std = df["Volume"].rolling(window=20).std()
    df["volume_zscore"] = (df["Volume"] - rolling_vol_mean) / (rolling_vol_std + 1e-8)
    
    df["realized_vol_20d"] = df["log_return"].rolling(window=20).std() * np.sqrt(252)
    
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))
   
    df["rsi_norm"] = (df["rsi_14"] - 50.0) / 50.0

    df["fwd_5d_return"] = (df["Close"].shift(-5) - df["Close"]) / df["Close"]
    
    conditions = [
        (df["fwd_5d_return"] > 0.02),
        (df["fwd_5d_return"] < -0.02)
    ]
    choices = [1, -1]
    df["target_class"] = np.select(conditions, choices, default=0)
    
    df = df.dropna().reset_index()
    return df

def fetch_and_process_universe():
    print(f"Fetching market data for: {TICKERS}")
    combined_records = []
    
    for ticker in TICKERS:
        print(f"  -> Downloading {ticker}...")
        raw_df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
            
        processed_df = compute_technical_indicators(raw_df)
        processed_df["ticker"] = ticker
        
        processed_df.to_parquet(f"{OUTPUT_DIR}/{ticker}.parquet")
        combined_records.append(processed_df)
        
    full_df = pd.concat(combined_records, ignore_index=True)
    full_df.to_parquet(f"{OUTPUT_DIR}/all_market_data.parquet")
    print(f"\n✅ Market Data successfully processed and saved to '{OUTPUT_DIR}/all_market_data.parquet'")
    print(f"Total Rows: {len(full_df)} across {len(TICKERS)} tickers.")

if __name__ == "__main__":
    fetch_and_process_universe()