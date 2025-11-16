#!/usr/bin/env python
# coding: utf-8

# In[4]:


# ------------------------------------------------------------------------------
#                            Data Collection
# ------------------------------------------------------------------------------

"""
Utilities to download and validate OHLCV market data.

• Uses yfinance (free) so you can start immediately.
• Works for US tickers (e.g., "AAPL") and NSE tickers (append ".NS", e.g., "RELIANCE.NS").
• Returns a clean DataFrame with: Open, High, Low, Close, Adj Close, Volume
"""

from __future__ import annotations
from typing import Iterable, Dict, Optional
import pandas as pd
import yfinance as yf


# --------------- Core Downloader ----------------
def get_price_data(
    ticker: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV for a single ticker and return a CLEAN DataFrame.

    Returns DataFrame indexed by date with columns:
    ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    """
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}. Check symbol or data range.")

    # ------------------------------------------------------------------
    # Robust normalization of column names (handles MultiIndex or tuple)
    # ------------------------------------------------------------------
    # If columns are a MultiIndex, prefer the first level (commonly "Open","Close",...)
    if isinstance(df.columns, pd.MultiIndex):
        cols = list(df.columns.get_level_values(0))
    else:
        cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                # pick first element if tuple-like, else join elements
                if len(c) > 0:
                    cols.append(str(c[0]))
                else:
                    cols.append(str(c))
            else:
                cols.append(str(c))

    # Title-case and strip to standardize: "adj close" -> "Adj Close"
    cols = [c.title().strip() for c in cols]
    df.columns = cols

    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Helpful debug: include returned columns in the error
        raise ValueError(f"{ticker}: Missing columns from yfinance: {missing}. Returned columns: {list(df.columns)}")

    # Keep only required columns in standard order
    df = df[required].copy()
    df.index = pd.to_datetime(df.index)

    # Clean and return
    df = _clean_prices(df)
    return df


# ----------- Batch helper for multiple tickers ------------------
def get_multi_price_data(
    tickers: Iterable[str],
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV for many tickers and return a dict[ticker] -> DataFrame.
    Skips/prints tickers that fail instead of stopping the whole run.
    """
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            out[t] = get_price_data(
                t, start=start, end=end, interval=interval, auto_adjust=auto_adjust
            )
        except Exception as e:
            print(f"[WARN] {t}: {e}")

    if not out:
        raise RuntimeError("No data collected for any ticker.")

    return out


# -------------- Convenience: save/load ------------------------
def save_to_csv(df: pd.DataFrame, path: str) -> None:
    """Save a single ticker DataFrame to CSV."""
    df.to_csv(path)


def load_from_csv(path: str) -> pd.DataFrame:
    """Load a CSV created by save_to_csv() and validate/clean it."""
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df.index = pd.to_datetime(df.index)
    return _clean_prices(df[required].copy())


# ------------------ Cleaner ------------------
def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, safe cleaning for price data:
    - sort by date and drop duplicates
    - coerce numeric dtypes
    - drop rows with NA (avoid filling to prevent leakage)
    - remove non-positive prices and negative volumes
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Coerce numeric types
    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # Build a single boolean mask to avoid repeated filtering side-effects
    positive_prices_mask = (df[price_cols] > 0).all(axis=1)
    non_negative_vol_mask = (df["Volume"] >= 0)

    df = df[positive_prices_mask & non_negative_vol_mask]

    # Drop any remaining NaNs
    df = df.dropna(how="any")

    if len(df) < 100:
        print("[INFO] Very short history (<100 rows). Modeling may be unreliable.")

    return df


# In[5]:


# Single ticker
df = get_price_data("RELIANCE.NS", start="2022-01-01")
print(df.head())
print(df.tail())

# Multiple tickers
tickers = ["TCS.NS", "INFY.NS", "HDFCBANK.NS"]
data = get_multi_price_data(tickers, start="2023-01-01")

# Save and reload
save_to_csv(df, "reliance.csv")
loaded_df = load_from_csv("reliance.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




