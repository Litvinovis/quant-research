#!/usr/bin/env python3
"""
"After N red candles in a row, sell (short); after N green candles in a
row, buy" — swept across candle timeframe, streak length, and hold
duration, all on real intraday data.

HoldingPeriodExit is bar-count based, not calendar-based (see engine.py),
so "hold for 10 minutes" on 5-minute candles is just holding_period(days=2)
— no engine changes needed for that part, only for the new
ConsecutiveCandlesCondition and the tz-aware intraday index (see
engine.py::_filter_by_date_range).

Scope note: intraday history is expensive to fetch and reason about at
"a year" of 1-minute bars for 15 tickers, so this uses a shorter, honestly
-stated window per timeframe rather than pretending to a full year.

Usage:
    .venv/bin/python scripts/candle_streak_sweep.py                  # 0.3% commission (default)
    .venv/bin/python scripts/candle_streak_sweep.py --commission 0.0004  # e.g. Tinkoff Premium tariff
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from tinkoff.invest import CandleInterval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from data import fetch_intraday_universe  # noqa: E402
from engine import run_backtest  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from run_log import log_run  # noqa: E402
from strategy_spec import StrategySpec  # noqa: E402

UNIVERSE = [
    "SBER", "GAZP", "LKOH", "GMKN", "NVTK", "ROSN", "TATN", "MTSS", "MGNT",
    "CHMF", "PLZL", "ALRS", "VTBR", "RUAL", "AFLT",
]
POSITION_WEIGHT = 0.10
DEFAULT_COMMISSION_PCT = 0.003

TIMEFRAMES = [
    # (label, CandleInterval, minutes-per-bar, lookback-days) —
    # 1-min bars are ~5x the data volume of 5-min, so a shorter window.
    ("1min", CandleInterval.CANDLE_INTERVAL_1_MIN, 1, 30),
    ("5min", CandleInterval.CANDLE_INTERVAL_5_MIN, 5, 180),
]
STREAK_COUNTS = [1, 2, 3]
HOLD_MINUTES = [5, 10]


def build_specs(
    timeframe_label: str, minutes_per_bar: int, start: date, end: date,
    universe: list[str], commission_pct: float, name_suffix: str,
) -> list[StrategySpec]:
    specs = []
    common = dict(
        universe=universe, start_date=start, end_date=end,
        commission_pct=commission_pct,
        position_sizing={"type": "equal_weight", "weight": POSITION_WEIGHT},
    )
    for count in STREAK_COUNTS:
        for hold_min in HOLD_MINUTES:
            hold_bars = max(1, round(hold_min / minutes_per_bar))
            specs.append(StrategySpec(
                name=f"streak_{timeframe_label}_green{count}_hold{hold_min}m_long{name_suffix}",
                entry={"type": "consecutive_candles", "count": count, "candle_color": "green"},
                exit={"type": "holding_period", "days": hold_bars},
                direction="long", **common,
            ))
            specs.append(StrategySpec(
                name=f"streak_{timeframe_label}_red{count}_hold{hold_min}m_short{name_suffix}",
                entry={"type": "consecutive_candles", "count": count, "candle_color": "red"},
                exit={"type": "holding_period", "days": hold_bars},
                direction="short", **common,
            ))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--commission", type=float, default=DEFAULT_COMMISSION_PCT,
        help="per-leg commission as a fraction, e.g. 0.0004 = 0.04%% (Tinkoff Premium tariff)",
    )
    args = parser.parse_args()
    name_suffix = "" if args.commission == DEFAULT_COMMISSION_PCT else f"_comm{args.commission * 100:.2f}pct".replace(".", "")
    print(f"Commission: {args.commission * 100:.3f}% per leg ({args.commission * 200:.3f}% round trip)")

    all_rows = []
    for label, interval, minutes_per_bar, lookback_days in TIMEFRAMES:
        start = date.today() - timedelta(days=lookback_days)
        end = date.today()
        print(f"\n--- {label} bars, {start} .. {end} ({lookback_days}d), {len(UNIVERSE)} tickers ---")
        data = fetch_intraday_universe(UNIVERSE, start, end, interval)
        resolved = sorted(data.keys())
        print(f"Resolved {len(resolved)}/{len(UNIVERSE)}: {resolved}")
        if not resolved:
            print(f"  no tickers resolved for {label}, skipping this timeframe")
            continue

        specs = build_specs(label, minutes_per_bar, start, end, resolved, args.commission, name_suffix)
        for spec in specs:
            result = run_backtest(spec, data)
            metrics = compute_metrics(result, spec.initial_capital)
            log_run(spec.model_dump(mode="json"), metrics)
            all_rows.append({"name": spec.name, "timeframe": label, **metrics})
            ret = metrics["total_return_pct"] if metrics["total_return_pct"] is not None else 0
            sharpe = metrics["sharpe"] if metrics["sharpe"] is not None else 0
            wr = metrics["win_rate_pct"] if metrics["win_rate_pct"] is not None else 0
            gross = metrics["total_return_pct_gross"] if metrics["total_return_pct_gross"] is not None else 0
            print(f"  {spec.name:52s} trades={metrics['num_trades']:5d}  "
                  f"net={ret:7.2f}%  gross={gross:7.2f}%  sharpe={sharpe:6.2f}  win_rate={wr:6.2f}%")

    print("\n=== Ranked by Sharpe (min 30 trades for significance) ===")
    ranked = sorted(
        (r for r in all_rows if r["num_trades"] >= 30 and r["sharpe"] is not None),
        key=lambda r: r["sharpe"], reverse=True,
    )
    for r in ranked[:10]:
        print(f"{r['name']:52s} [{r['timeframe']}] trades={r['num_trades']:5d}  "
              f"net={r['total_return_pct']:7.2f}%  gross={r['total_return_pct_gross']:7.2f}%  "
              f"sharpe={r['sharpe']:6.2f}  win_rate={r['win_rate_pct']:6.2f}%  max_dd={r['max_drawdown_pct']:7.2f}%")

    if not ranked:
        print("(none reached 30 trades — showing all instead)")
        for r in sorted(all_rows, key=lambda r: r["sharpe"] or -999, reverse=True)[:10]:
            print(f"{r['name']:52s} [{r['timeframe']}] trades={r['num_trades']:5d}  "
                  f"net={r['total_return_pct']}%  sharpe={r['sharpe']}")


if __name__ == "__main__":
    main()
