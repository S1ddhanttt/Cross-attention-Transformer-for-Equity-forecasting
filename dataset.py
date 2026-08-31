import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class StockRAGDataset(Dataset):
    def __init__(
        self,
        market_parquet_path: str = "data/processed_market/all_market_data.parquet",
        sec_embed_parquet_path: str = "data/processed_text/sec_embeddings.parquet",
        seq_len: int = 60,
        k_filings: int = 3,
        embed_dim: int = 384,
        feature_cols: list = None
    ):
        self.seq_len = seq_len
        self.k_filings = k_filings
        self.embed_dim = embed_dim
        
        if feature_cols is None:
            self.feature_cols = [
                "log_return",
                "normalized_range",
                "normalized_body",
                "volume_zscore",
                "realized_vol_20d",
                "rsi_norm"
            ]
        else:
            self.feature_cols = feature_cols

        # 1. Load Data
        print("Loading market and textual embedding tables...")
        self.df_market = pd.read_parquet(market_parquet_path)
        self.df_sec = pd.read_parquet(sec_embed_parquet_path)
        
        # Ensure proper datetime types
        self.df_market["Date"] = pd.to_datetime(self.df_market["Date"])
        self.df_sec["filing_date"] = pd.to_datetime(self.df_sec["filing_date"])
        
        # Map target classes [-1, 0, 1] to [0, 1, 2] for PyTorch CrossEntropyLoss
        target_map = {-1: 0, 0: 1, 1: 2}
        self.df_market["target_idx"] = self.df_market["target_class"].map(target_map)

        # 2. Build sample indices (ticker + valid index with 60 days history)
        self.samples = []
        for ticker, group in self.df_market.groupby("ticker"):
            group = group.sort_values("Date").reset_index(drop=True)
            for i in range(self.seq_len, len(group)):
                trade_date = group.loc[i, "Date"]
                target = group.loc[i, "target_idx"]
                price_window = group.loc[i - self.seq_len : i - 1, self.feature_cols].values
                self.samples.append((ticker, trade_date, price_window, target))
                
        print(f"✅ Prepared {len(self.samples)} aligned samples across all tickers.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker, trade_date, price_window, target = self.samples[idx]
        
        # --- Point-in-Time (PIT) Text Retrieval ---
        # Find filings for this ticker published on or before trade_date within last 60 days
        cutoff_date = trade_date - pd.Timedelta(days=60)
        recent_filings = self.df_sec[
            (self.df_sec["ticker"] == ticker) &
            (self.df_sec["filing_date"] <= trade_date) &
            (self.df_sec["filing_date"] >= cutoff_date)
        ].sort_values("filing_date", ascending=False)

        # Extract top-K embeddings
        if len(recent_filings) > 0:
            raw_embs = list(recent_filings["embedding"].iloc[: self.k_filings])
            # If fewer than K, pad with zero vectors
            while len(raw_embs) < self.k_filings:
                raw_embs.append(np.zeros(self.embed_dim, dtype=np.float32))
            text_matrix = np.array(raw_embs, dtype=np.float32)
        else:
            # Neutral pad matrix if no news in window
            text_matrix = np.zeros((self.k_filings, self.embed_dim), dtype=np.float32)

        price_tensor = torch.tensor(price_window, dtype=torch.float32)   # [60, num_features]
        text_tensor = torch.tensor(text_matrix, dtype=torch.float32)     # [K, 384]
        target_tensor = torch.tensor(target, dtype=torch.long)           # scalar

        return price_tensor, text_tensor, target_tensor

# Test DataLoader execution
if __name__ == "__main__":
    dataset = StockRAGDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Inspect a single batch
    prices, text_embs, targets = next(iter(loader))
    print(f"\nBatch Shapes:")
    print(f"  • Price Tensor:       {prices.shape}      -> [Batch, 60_days, 6_features]")
    print(f"  • RAG Embeddings:     {text_embs.shape}   -> [Batch, 3_filings, 384_dim]")
    print(f"  • Target Class:       {targets.shape}     -> [Batch]")