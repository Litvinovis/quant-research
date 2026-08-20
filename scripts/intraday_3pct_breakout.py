#!/usr/bin/env python3
"""
One-off analysis (not part of the general StrategySpec DSL — that DSL is
daily-bar only, this rule is inherently intraday):

  Blue-chip MOEX stocks. First intraday bar where price is +3% vs
  PREVIOUS DAY'S CLOSE -> buy at the NEXT bar's open (no lookahead).
  Sell at the LAST intraday bar of the SAME trading day (proxy for
  "before session close"). Commission 0.3% per leg (0.6% round trip).

Uses 5-minute candles. Not cached (deliberately — this is a one-shot
analysis script, not the reusable data.py pipeline).
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tinkoff.invest import CandleInterval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import _client, _quotation_to_float  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BLUE_CHIPS = ["SBER", "GAZP", "LKOH", "GMKN", "NVTK", "ROSN", "TATN", "MTSS", "MGNT", "CHMF"]
BREAKOUT_PCT = 0.03
COMMISSION_PCT = 0.003  # per leg, per the user's spec
LOOKBACK_DAYS = 120  # ~4 months of 5-min bars is already a lot of API calls


def fetch_intraday(services, ticker: str, start: date, end: date) -> pd.DataFrame:
    result = services.instruments.find_instrument(query=ticker)
    figi = None
    for instrument in result.instruments:
        if instrument.ticker == ticker and instrument.class_code in ("TQBR", "SPBXM"):
            figi = instrument.figi
            break
    if figi is None:
        for instrument in result.instruments:
            if instrument.ticker == ticker:
                figi = instrument.figi
                break
    if figi is None:
        print(f"  {ticker}: not found, skipping")
        return pd.DataFrame()

    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc)
    rows = []
    for candle in services.get_all_candles(
        figi=figi, from_=start_dt, to=end_dt, interval=CandleInterval.CANDLE_INTERVAL_5_MIN
    ):
        rows.append(
            {
                "ts": candle.time,
                "open": _quotation_to_float(candle.open),
                "close": _quotation_to_float(candle.close),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("ts")
    df["day"] = df["ts"].dt.date
    return df


def simulate(df: pd.DataFrame, ticker: str) -> list[dict]:
    trades = []
    days = df["day"].unique()
    prev_close = None
    for day in days:
        day_df = df[df["day"] == day].reset_index(drop=True)
        if prev_close is None:
            prev_close = day_df["close"].iloc[-1]
            continue

        entry_i = None
        for i in range(len(day_df)):
            change = (day_df["close"].iloc[i] - prev_close) / prev_close
            if change >= BREAKOUT_PCT:
                entry_i = i
                break

        if entry_i is not None and entry_i + 1 < len(day_df):
            entry_price = day_df["open"].iloc[entry_i + 1]
            exit_price = day_df["close"].iloc[-1]
            gross_pct = (exit_price - entry_price) / entry_price
            net_pct = gross_pct - 2 * COMMISSION_PCT
            trades.append(
                {
                    "ticker": ticker,
                    "day": day,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": net_pct,
                }
            )

        prev_close = day_df["close"].iloc[-1]

    return trades


def main() -> None:
    end = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    print(f"Period: {start} .. {end}, {len(BLUE_CHIPS)} tickers, 5-min bars")

    all_trades = []
    with _client(os.environ["TINKOFF_TOKEN"]) as services:
        for ticker in BLUE_CHIPS:
            print(f"fetching {ticker}...")
            df = fetch_intraday(services, ticker, start, end)
            if df.empty:
                continue
            trades = simulate(df, ticker)
            print(f"  {ticker}: {len(trades)} trades")
            all_trades.extend(trades)

    if not all_trades:
        print("No trades triggered — the +3% intraday breakout never fired in this window.")
        return

    pnls = [t["pnl_pct"] for t in all_trades]
    equity = 1_000_000.0
    curve = [equity]
    for p in pnls:
        equity *= 1 + p
        curve.append(equity)
    curve = pd.Series(curve)
    running_max = curve.cummax()
    max_dd = ((curve - running_max) / running_max).min()

    print()
    print("=== Results ===")
    print(f"trades: {len(all_trades)}")
    print(f"win rate: {sum(1 for p in pnls if p > 0) / len(pnls) * 100:.2f}%")
    print(f"avg trade pnl (net of commission): {np.mean(pnls) * 100:.3f}%")
    print(f"total return: {(curve.iloc[-1] / curve.iloc[0] - 1) * 100:.2f}%")
    print(f"max drawdown: {max_dd * 100:.2f}%")
    print()
    print("by ticker:")
    by_ticker = pd.DataFrame(all_trades).groupby("ticker")["pnl_pct"].agg(["count", "mean", "sum"])
    by_ticker.columns = ["trades", "avg_pnl_pct", "sum_pnl_pct"]
    by_ticker["avg_pnl_pct"] *= 100
    by_ticker["sum_pnl_pct"] *= 100
    print(by_ticker.round(3))


if __name__ == "__main__":
    main()
