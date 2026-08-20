"""Deterministic indicator functions, all pure pandas — no LLM in the loop."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pct_change(series: pd.Series, lookback_days: int) -> pd.Series:
    return series.pct_change(periods=lookback_days)


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    # Standard RSI formula divides avg_gain/avg_loss, which is undefined when
    # avg_loss is 0 (no losses in the window). By convention that's RSI=100
    # if there were any gains, or RSI=50 for a perfectly flat window (no
    # gains and no losses) — not NaN, which a naive division would produce.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    no_loss = avg_loss == 0
    result = result.where(~no_loss, np.where(avg_gain > 0, 100.0, 50.0))
    return result
