import os
import torch
import numpy as np
import pandas as pd
import vectorbt as vbt
from torch.utils.data import DataLoader, Subset

from dataset import StockRAGDataset
from model import MultimodalStockTransformer

# Configuration
CHECKPOINT_PATH = "checkpoints/best_multimodal_transformer.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEES = 0.001  # 10 bps (0.1%) transaction fee per trade (slippage + commission)

def run_backtest():
    # 1. Load Dataset & Validation Split (Out-of-Sample)
    dataset = StockRAGDataset()
    total_samples = len(dataset)
    split_idx = int(0.8 * total_samples)
    val_indices = list(range(split_idx, total_samples))
    
    val_set = Subset(dataset, val_indices)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # 2. Load Model Checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at '{CHECKPOINT_PATH}'. Run train.py first.")
        
    model = MultimodalStockTransformer(
        price_features=6,
        rag_dim=384,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_classes=3
    ).to(DEVICE)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from Epoch {checkpoint['epoch']} with Val F1: {checkpoint['val_f1']:.4f}")

    # 3. Model Inference (Probabilities for Bullish vs Bearish)
    all_bullish_probs = []
    all_bearish_probs = []
    
    with torch.no_grad():
        for price_b, text_b, _ in val_loader:
            price_b = price_b.to(DEVICE)
            text_b = text_b.to(DEVICE)
            logits, _ = model(price_b, text_b)
            
            # Convert logits to probabilities (Softmax)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            # Class 0: Bearish, Class 1: Neutral, Class 2: Bullish
            all_bearish_probs.extend(probs[:, 0])
            all_bullish_probs.extend(probs[:, 2])

    # 4. Build Multi-Asset Validation Panel
    val_records = []
    for i, idx in enumerate(val_indices):
        ticker, trade_date, _, _ = dataset.samples[idx]
        bull_p = all_bullish_probs[i]
        bear_p = all_bearish_probs[i]
        
        # Raw Alpha Signal = P(Bullish) - P(Bearish) (Ranges from -1.0 to +1.0)
        signal = bull_p - bear_p
        
        val_records.append({
            "Date": trade_date,
            "ticker": ticker,
            "signal": signal
        })

    df_signals = pd.DataFrame(val_records)
    
    # Pivot signals and actual market prices into matrix: [Dates x Tickers]
    df_market = dataset.df_market.copy()
    df_close = df_market.pivot(index="Date", columns="ticker", values="Close").ffill()
    df_signal_matrix = df_signals.pivot(index="Date", columns="ticker", values="signal").reindex(df_close.index)

    # Filter for the validation date range only
    val_dates = df_signals["Date"].unique()
    min_date, max_date = val_dates.min(), val_dates.max()
    df_close_val = df_close.loc[min_date:max_date]
    df_signal_val = df_signal_matrix.loc[min_date:max_date].fillna(0.0)

    # 5. Long/Short Dollar-Neutral Strategy Rules
    # Top 33% positive signal -> Long (+1) | Bottom 33% negative signal -> Short (-1)
    # Rebalance weekly (every 5 trading days)
    target_allocations = pd.DataFrame(0.0, index=df_signal_val.index, columns=df_signal_val.columns)

    for step_idx in range(0, len(df_signal_val), 5):
        current_date = df_signal_val.index[step_idx]
        scores = df_signal_val.loc[current_date]
        
        # Rank assets cross-sectionally
        n_assets = len(scores)
        top_k = max(1, n_assets // 3)
        
        ranked = scores.sort_values(ascending=False)
        long_tickers = ranked.iloc[:top_k].index
        short_tickers = ranked.iloc[-top_k:].index
        
        # Long/Short equal weights normalized to 100% gross exposure
        weight_per_long = 0.5 / len(long_tickers)
        weight_per_short = -0.5 / len(short_tickers)
        
        # Hold positions for 5 trading days
        end_step = min(step_idx + 5, len(df_signal_val))
        target_allocations.iloc[step_idx:end_step][long_tickers] = weight_per_long
        target_allocations.iloc[step_idx:end_step][short_tickers] = weight_per_short

    # 6. Run VectorBT Simulation
    print("\nRunning VectorBT Backtest...")
    portfolio = vbt.Portfolio.from_orders(
        close=df_close_val,
        size=target_allocations,
        size_type="targetpercent",
        cash_sharing=True,
        group_by=True,  # Combines into a single unified multi-asset portfolio
        init_cash=100_000.0,
        fees=FEES,
        freq="1D"
    )

    # 7. Print Performance Metrics
    stats = portfolio.stats()
    print("\n" + "="*50)
    print("      QUANTITATIVE STRATEGY PERFORMANCE REPORT")
    print("="*50)
    print(f"Start Date:               {min_date.strftime('%Y-%m-%d')}")
    print(f"End Date:                 {max_date.strftime('%Y-%m-%d')}")
    print(f"Total Return:             {portfolio.total_return() * 100:.2f}%")
    print(f"Benchmark Return (Hold):  {portfolio.benchmark_return() * 100:.2f}%")
    print(f"Annualized Return:        {portfolio.annualized_return() * 100:.2f}%")
    print(f"Annualized Volatility:    {portfolio.annualized_volatility() * 100:.2f}%")
    print(f"Sharpe Ratio:             {portfolio.sharpe_ratio():.3f}")
    print(f"Sortino Ratio:            {portfolio.sortino_ratio():.3f}")
    print(f"Calmar Ratio:             {portfolio.calmar_ratio():.3f}")
    print(f"Max Drawdown:             {portfolio.max_drawdown() * 100:.2f}%")
    print(f"Max Drawdown Duration:    {portfolio.max_drawdown_duration()}")
    print(f"Total Trades Executed:    {portfolio.total_trades()}")
    print("="*50)

    # 8. Save Equity Curve & Drawdown Charts
    try:
        fig = portfolio.plot(subplots=["cum_returns", "drawdowns"])
        fig.write_image("backtest_performance.png")
        print("Performance plots saved to 'backtest_performance.png'.")
    except Exception:
        # Fallback if plotly kaleido is not installed
        pass

if __name__ == "__main__":
    run_backtest()