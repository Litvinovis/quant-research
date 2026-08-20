"""Performance metrics computed from a BacktestResult's equity curve/trades."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import BacktestResult


def _total_return_and_cagr(curve: pd.Series, initial_capital: float) -> tuple[float, float | None]:
    total_return = (curve.iloc[-1] - initial_capital) / initial_capital
    days_elapsed = (curve.index[-1] - curve.index[0]).days
    years = days_elapsed / 365.25
    cagr = (curve.iloc[-1] / initial_capital) ** (1 / years) - 1 if years > 0 else None
    return total_return, cagr


def compute_metrics(result: BacktestResult, initial_capital: float) -> dict:
    curve = result.equity_curve
    trades = result.trades

    if len(curve) < 2 or not trades:
        return {
            "num_trades": len(trades),
            "total_return_pct": 0.0,
            "total_return_pct_gross": 0.0,
            "cagr_pct": None,
            "cagr_pct_gross": None,
            "max_drawdown_pct": 0.0,
            "sharpe": None,
            "win_rate_pct": None,
            "avg_trade_pnl_pct": None,
        }

    total_return, cagr = _total_return_and_cagr(curve, initial_capital)
    total_return_gross, cagr_gross = _total_return_and_cagr(result.gross_equity_curve, initial_capital)

    running_max = curve.cummax()
    drawdown = (curve - running_max) / running_max
    max_drawdown = drawdown.min()

    daily = curve.resample("D").last().ffill().pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else None

    pnls = [t.net_pnl_pct for t in trades]
    win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
    avg_trade_pnl = float(np.mean(pnls))

    return {
        "num_trades": len(trades),
        "total_return_pct": round(total_return * 100, 2),
        "total_return_pct_gross": round(total_return_gross * 100, 2),
        "cagr_pct": round(cagr * 100, 2) if cagr is not None else None,
        "cagr_pct_gross": round(cagr_gross * 100, 2) if cagr_gross is not None else None,
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_trade_pnl_pct": round(avg_trade_pnl * 100, 2),
    }
