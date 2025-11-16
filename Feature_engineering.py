#!/usr/bin/env python
# coding: utf-8

# In[3]:


# get_ipython().system('pip install openpyxl')


# In[ ]:


#   --------------------------------------------------------------------------------------
#                        Feature_Engineering.py
#   --------------------------------------------------------------------------------------


"""
    Turns raw OHLCV into ML-ready features + targets.


    Design goals:
    • Purely backward-looking (no lookahead leakage).
    • Stable defaults that work even if the `ta` library isn't installed.
    • Both classification (up/down) and regression (next-return) targets.

    Outputs:
    - add_indicators(df): returns df with engineered columns.
    - make_classification_target(df, horizon=1): adds binary 'target' (1 if next return > 0).
    - make_regression_target(df, horizon=1): adds 'y_reg' = next log-return.
    - build_dataset(df): returns (X, y_cls, y_reg, feature_columns, df_aligned)
"""


# In[1]:


import numpy as np
import pandas as pd
from typing import Optional, Sequence, Tuple, Dict

# Optional 'ta' library for technical indicators
try:
    from ta.trend import SMAIndicator, EMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator
    TA_AVAILABLE = True
except Exception:
    TA_AVAILABLE = False


# --------------------------------------------------
# Configuration
# --------------------------------------------------
DEFAULT_FEATURES = [
    "ret_1", "ret_3", "ret_5",
    "log_close",
    "sma_10", "sma_50", "ema_12",
    "macd", "rsi_14",
    "bb_h", "bb_l",
    "atr_14", "obv",
    "sma10_minus_50",
    "price_bb_pct",
    "vol_10",
    "vol_ratio_20", "ret_1_lag1", "ret_1_lag2", "rolling_ret_10_mean"
]

_REQUIRED_COLS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}


def _validate_input(df: pd.DataFrame):
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")


def _zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std(ddof=0) + 1e-12)


# --------------------------------------------------
# Feature Builder Class
# --------------------------------------------------
class FeatureBuilder:
    def __init__(
        self,
        sma_windows: Sequence[int] = (10, 50),
        ema_window: int = 12,
        rsi_window: int = 14,
        bb_window: int = 20,
        bb_dev: float = 2.0,
        atr_window: int = 14,
        vol_ratio_window: int = 20,
        min_periods: Optional[int] = None,
        dropna: bool = True,
        normalize: bool = False,
        feature_list: Optional[Sequence[str]] = None,
    ):
        self.sma_windows = list(sma_windows)
        self.ema_window = ema_window
        self.rsi_window = rsi_window
        self.bb_window = bb_window
        self.bb_dev = bb_dev
        self.atr_window = atr_window
        self.vol_ratio_window = vol_ratio_window
        self.min_periods = min_periods
        self.dropna = dropna
        self.normalize = normalize
        self.feature_list = list(feature_list) if feature_list else DEFAULT_FEATURES


    def _mp(self, w: int) -> int:
        """
        Helper to decide min_periods for rolling operations:
        - if self.min_periods is set (not None) use that,
        - otherwise default to the window size `w`.
        """
        return int(self.min_periods) if (self.min_periods is not None) else int(w)

    # --------------------------------------------------
    # Step 1: Add technical indicators
    # --------------------------------------------------
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        _validate_input(df)
        out = df.copy().astype(float)

        # Basic returns and log close
        out["ret_1"] = out["Adj Close"].pct_change(1)
        out["ret_3"] = out["Adj Close"].pct_change(3)
        out["ret_5"] = out["Adj Close"].pct_change(5)
        out["log_close"] = np.log(out["Adj Close"])

        # Simple Moving Averages
        for w in self.sma_windows:
            col = f"sma_{w}"
            if TA_AVAILABLE:
                out[col] = SMAIndicator(out["Adj Close"], window=w).sma_indicator()
            else:
                out[col] = out["Adj Close"].rolling(window=w, min_periods=self.min_periods or w).mean()

        out["next_open"] = out["Open"].shift(-1)
        out["next_close"] = out["Close"].shift(-1)


        # EMA
        if TA_AVAILABLE:
            out[f"ema_{self.ema_window}"] = EMAIndicator(out["Adj Close"], window=self.ema_window).ema_indicator()
        else:
            out[f"ema_{self.ema_window}"] = out["Adj Close"].ewm(span=self.ema_window, adjust=False).mean()

        # MACD
        if TA_AVAILABLE:
            out["macd"] = MACD(out["Adj Close"]).macd()
        else:
            ema12 = out["Adj Close"].ewm(span=12, adjust=False).mean()
            ema26 = out["Adj Close"].ewm(span=26, adjust=False).mean()
            out["macd"] = ema12 - ema26

        # RSI
        if TA_AVAILABLE:
            out[f"rsi_{self.rsi_window}"] = RSIIndicator(out["Adj Close"], window=self.rsi_window).rsi()
        else:
            delta = out["Adj Close"].diff()
            up = delta.clip(lower=0).rolling(self.rsi_window).mean()
            down = (-delta.clip(upper=0)).rolling(self.rsi_window).mean()
            rs = up / (down.replace(0, np.nan))
            out[f"rsi_{self.rsi_window}"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        if TA_AVAILABLE:
            bb = BollingerBands(close=out["Adj Close"], window=self.bb_window, window_dev=self.bb_dev)
            out["bb_h"] = bb.bollinger_hband()
            out["bb_l"] = bb.bollinger_lband()
        else:
            ma = out["Adj Close"].rolling(self.bb_window).mean()
            sd = out["Adj Close"].rolling(self.bb_window).std()
            out["bb_h"] = ma + self.bb_dev * sd
            out["bb_l"] = ma - self.bb_dev * sd

        # ATR
        if TA_AVAILABLE:
            out[f"atr_{self.atr_window}"] = AverageTrueRange(
                high=out["High"], low=out["Low"], close=out["Close"], window=self.atr_window
            ).average_true_range()
        else:
            tr = pd.concat([
                out["High"] - out["Low"],
                (out["High"] - out["Close"].shift()).abs(),
                (out["Low"] - out["Close"].shift()).abs()
            ], axis=1).max(axis=1)
            out[f"atr_{self.atr_window}"] = tr.rolling(self.atr_window).mean()

        # OBV
        if TA_AVAILABLE:
            out["obv"] = OnBalanceVolumeIndicator(close=out["Adj Close"], volume=out["Volume"]).on_balance_volume()
        else:
            obv = []
            prev_obv = 0.0
            prev_close = np.nan
            for idx, (close, vol) in enumerate(zip(out["Adj Close"].values, out["Volume"].values)):
                if idx == 0:
                    obv.append(0.0)
                    prev_close = close
                    prev_obv = 0.0
                    continue
                if np.isnan(close) or np.isnan(prev_close):
                    obv.append(prev_obv)
                elif close > prev_close:
                    prev_obv = prev_obv + vol
                    obv.append(prev_obv)
                elif close < prev_close:
                    prev_obv = prev_obv - vol
                    obv.append(prev_obv)
                else:
                    obv.append(prev_obv)
                prev_close = close
            out["obv"] = pd.Series(obv, index=out.index)
        # Derived features
        if len(self.sma_windows) >= 2:
            short, long = self.sma_windows[0], self.sma_windows[1]
            diff_col = f"sma_{short}_minus_{long}"
            out[diff_col] = out[f"sma_{short}"] - out[f"sma_{long}"]
            # compatibility alias
            if short == 10 and long == 50:
                out["sma10_minus_50"] = out[diff_col]

        # price bollinger percent, safe division
        denom = (out["bb_h"] - out["bb_l"]).replace(0, np.nan)
        out["price_bb_pct"] = (out["Adj Close"] - out["bb_l"]) / (denom + 1e-12)

        # volume/volatility features
        out["vol_10"] = out["ret_1"].rolling(10, min_periods=self._mp(10)).std()
        out["vol_ratio_20"] = out["Volume"] / (out["Volume"].rolling(self.vol_ratio_window, min_periods=self._mp(self.vol_ratio_window)).mean() + 1e-12)
        out["ret_1_lag1"] = out["ret_1"].shift(1)
        out["ret_1_lag2"] = out["ret_1"].shift(2)
        out["rolling_ret_10_mean"] = out["ret_1"].rolling(10, min_periods=self._mp(10)).mean()

        if self.dropna:
            out = out.dropna()

        return out

    # --------------------------------------------------
    # Step 2: Create Targets
    # --------------------------------------------------
    def make_classification_target(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        threshold: float = 0.0,
        multi_class: bool = False,
        neutral_band: Optional[float] = None,
    ) -> pd.DataFrame:
        out = df.copy()
        future_ret = (out["Adj Close"].shift(-horizon) / out["Adj Close"]) - 1
        if not multi_class:
            out["target"] = (future_ret > threshold).astype(int)
        else:
            nb = neutral_band if neutral_band is not None else threshold
            out["target"] = np.where(future_ret > nb, 1, np.where(future_ret < -nb, -1, 0))
        return out.dropna(subset=["target"])

    def make_regression_target(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        out = df.copy()
        log_price = np.log(out["Adj Close"])
        out["y_reg"] = log_price.shift(-horizon) - log_price
        return out.dropna(subset=["y_reg"])

    # --------------------------------------------------
    # Step 3: Combine Features + Targets
    # --------------------------------------------------
    def build_dataset(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        classification_kwargs: Optional[Dict] = None,
        regression: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series], Sequence[str], pd.DataFrame]:
        classification_kwargs = classification_kwargs or {}
        fdf = self.add_indicators(df)
        cdf = self.make_classification_target(fdf, horizon=horizon, **classification_kwargs)
        rdf = self.make_regression_target(fdf, horizon=horizon) if regression else None

        idx = cdf.index.intersection(rdf.index) if rdf is not None else cdf.index
        aligned = fdf.loc[idx].copy()

        if "next_open" in fdf.columns:
            aligned["next_open"] = fdf.loc[idx, "next_open"]

        if "next_close" in fdf.columns:
            aligned["next_close"] = fdf.loc[idx, "next_close"]
            
        aligned["target"] = cdf.loc[idx, "target"]
        if rdf is not None:
            aligned["y_reg"] = rdf.loc[idx, "y_reg"]

        features = [f for f in self.feature_list if f in aligned.columns]
        X = aligned[features].copy()
        y_cls = aligned["target"].astype(int)
        y_reg = aligned["y_reg"].astype(float) if rdf is not None else None

        if self.normalize:
            X = X.apply(_zscore)

        return X, y_cls, y_reg, features, aligned



# In[4]:


from data_collection import get_price_data
#from Feature_engineering import FeatureBuilder

df = get_price_data("RELIANCE.NS", start="2016-01-01")
builder = FeatureBuilder(normalize=False, min_periods=5)
X, y_cls, y_reg, feats, aligned = builder.build_dataset(df, horizon=1)

print("Features:", feats)
print("X shape:", X.shape)
print("Label distribution:\n", y_cls.value_counts(normalize=True))
print(aligned.tail()[feats + ["target", "y_reg"]])

aligned.to_excel("reliance_features.xlsx",  index = True)
print("Data Saved to reliance_features.xlsx ✅")



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




