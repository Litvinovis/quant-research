#!/usr/bin/env python3
"""
Sweep: "big one-day move (proxy for positive news) continues 1-3 days" —
tests entry thresholds x holding periods x a couple of stop/take-profit
exit variants, on the same StrategySpec DSL/engine as everything else in
this repo. Each combo is logged to Postgres (backtest_runs) same as a
normal run.py invocation, so the sweep is browsable in Grafana too.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from data import fetch_universe  # noqa: E402
from engine import run_backtest  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from run_log import log_run  # noqa: E402
from strategy_spec import StrategySpec  # noqa: E402

UNIVERSE = [
    "SBER", "GAZP", "LKOH", "GMKN", "NVTK", "ROSN", "TATN", "MTSS", "MGNT",
    "CHMF", "PLZL", "ALRS", "VTBR", "RUAL", "AFLT",
]
COMMISSION_PCT = 0.003
START = date.today() - timedelta(days=730)
END = date.today()

POSITION_WEIGHT = 0.10  # 10% of current equity per trade — realistic diversification
                        # across 15 names, not "bet everything on every signal"
                        # (the engine's default weight=1.0 compounds unrealistically
                        # over hundreds of sequential trades — see README)
ENTRY_THRESHOLDS = [0.03, 0.05, 0.07]
HOLDING_DAYS = [1, 2, 3]
STOP_TAKE_VARIANTS = [
    {"stop_loss_pct": 0.03, "take_profit_pct": 0.05, "max_days": 3},
    {"stop_loss_pct": 0.03, "take_profit_pct": 0.08, "max_days": 3},
]


def build_specs() -> list[StrategySpec]:
    specs = []
    for threshold in ENTRY_THRESHOLDS:
        entry = {
            "type": "pct_change", "field": "close", "lookback_days": 1,
            "operator": ">", "value": threshold,
        }
        for days in HOLDING_DAYS:
            specs.append(StrategySpec(
                name=f"momentum_{int(threshold*100)}pct_hold{days}d",
                universe=UNIVERSE, start_date=START, end_date=END,
                entry=entry, exit={"type": "holding_period", "days": days},
                commission_pct=COMMISSION_PCT,
                position_sizing={"type": "equal_weight", "weight": POSITION_WEIGHT},
            ))
        for i, variant in enumerate(STOP_TAKE_VARIANTS):
            specs.append(StrategySpec(
                name=f"momentum_{int(threshold*100)}pct_sltp{i}",
                universe=UNIVERSE, start_date=START, end_date=END,
                entry=entry, exit={"type": "stop_loss_take_profit", **variant},
                commission_pct=COMMISSION_PCT,
                position_sizing={"type": "equal_weight", "weight": POSITION_WEIGHT},
            ))
    return specs


def main() -> None:
    print(f"Fetching {len(UNIVERSE)} tickers, {START} .. {END}...")
    data = fetch_universe(UNIVERSE, START, END)
    print("Data ready.\n")

    specs = build_specs()
    rows = []
    for spec in specs:
        result = run_backtest(spec, data)
        metrics = compute_metrics(result, spec.initial_capital)
        log_run(spec.model_dump(mode="json"), metrics)
        rows.append({"name": spec.name, **metrics})
        print(f"{spec.name:32s} trades={metrics['num_trades']:4d}  "
              f"return={metrics['total_return_pct'] if metrics['total_return_pct'] is not None else 0:7.2f}%  "
              f"sharpe={metrics['sharpe'] if metrics['sharpe'] is not None else 0:6.2f}  "
              f"win_rate={metrics['win_rate_pct'] if metrics['win_rate_pct'] is not None else 0:6.2f}%  "
              f"max_dd={metrics['max_drawdown_pct']:7.2f}%")

    print("\n=== Ranked by Sharpe (min 15 trades for significance) ===")
    ranked = sorted(
        (r for r in rows if r["num_trades"] >= 15 and r["sharpe"] is not None),
        key=lambda r: r["sharpe"], reverse=True,
    )
    for r in ranked[:5]:
        print(f"{r['name']:32s} trades={r['num_trades']:4d}  return={r['total_return_pct']:7.2f}%  "
              f"sharpe={r['sharpe']:6.2f}  win_rate={r['win_rate_pct']:6.2f}%  "
              f"cagr={r['cagr_pct']}%  max_dd={r['max_drawdown_pct']:7.2f}%")


if __name__ == "__main__":
    main()
