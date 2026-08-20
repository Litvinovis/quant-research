#!/usr/bin/env python3
"""
Mirror of momentum_continuation_sweep.py: instead of buying on a big
one-day rise (good news), short on a big one-day drop (bad news), same
holding-period / stop-loss-take-profit exit grid. Same blue-chip universe
and realistic 10% position sizing as the long sweep, for a fair
side-by-side comparison in the strategies catalog.
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
POSITION_WEIGHT = 0.10
START = date.today() - timedelta(days=730)
END = date.today()

DROP_THRESHOLDS = [0.03, 0.05, 0.07]
HOLDING_DAYS = [1, 2, 3]
# same magnitudes as the long sweep's stop/take variants — the engine now
# reinterprets them correctly for "short": stop_loss_pct = adverse RISE,
# take_profit_pct = favorable FALL.
STOP_TAKE_VARIANTS = [
    {"stop_loss_pct": 0.03, "take_profit_pct": 0.05, "max_days": 3},
    {"stop_loss_pct": 0.03, "take_profit_pct": 0.08, "max_days": 3},
]


def build_specs() -> list[StrategySpec]:
    specs = []
    for threshold in DROP_THRESHOLDS:
        entry = {
            "type": "pct_change", "field": "close", "lookback_days": 1,
            "operator": "<", "value": -threshold,
        }
        for days in HOLDING_DAYS:
            specs.append(StrategySpec(
                name=f"short_baddrop_{int(threshold*100)}pct_hold{days}d",
                universe=UNIVERSE, start_date=START, end_date=END,
                entry=entry, exit={"type": "holding_period", "days": days},
                direction="short", commission_pct=COMMISSION_PCT,
                position_sizing={"type": "equal_weight", "weight": POSITION_WEIGHT},
            ))
        for i, variant in enumerate(STOP_TAKE_VARIANTS):
            specs.append(StrategySpec(
                name=f"short_baddrop_{int(threshold*100)}pct_sltp{i}",
                universe=UNIVERSE, start_date=START, end_date=END,
                entry=entry, exit={"type": "stop_loss_take_profit", **variant},
                direction="short", commission_pct=COMMISSION_PCT,
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
        ret = metrics['total_return_pct'] if metrics['total_return_pct'] is not None else 0
        sharpe = metrics['sharpe'] if metrics['sharpe'] is not None else 0
        wr = metrics['win_rate_pct'] if metrics['win_rate_pct'] is not None else 0
        print(f"{spec.name:32s} trades={metrics['num_trades']:4d}  return={ret:7.2f}%  "
              f"sharpe={sharpe:6.2f}  win_rate={wr:6.2f}%  max_dd={metrics['max_drawdown_pct']:7.2f}%")

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
