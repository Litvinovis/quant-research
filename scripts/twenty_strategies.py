#!/usr/bin/env python3
"""
20 strategies built on genuinely different principles (not the same rule
with a different %), run on the same real blue-chip universe/period so
they're comparable in the catalog:

  1-4   trend-following via SMA crossover, several timeframes, long
  5-8   RSI mean-reversion at extremes, long AND short
  9-10  regime filter: price vs a long-term SMA (trade WITH the regime)
  11    momentum continuation on a big one-day move (yesterday's finding),
        with an RSI-based exit instead of stop/take, as a variant
  12-15 the OPPOSITE bet on a big one-day move: buy the dip / fade the
        spike (mean-reversion), long and short
  16-17 short-horizon scalps off a short SMA
  18    trend-following with an unbounded "let it run" condition exit
  19-20 mild (non-extreme) RSI mean-reversion, long and short

Same 15-ticker blue-chip universe and 2-year window as the momentum/short
sweeps, so this sits next to them in the catalog on equal footing.
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

COMMON = dict(universe=UNIVERSE, start_date=START, end_date=END,
              commission_pct=COMMISSION_PCT,
              position_sizing={"type": "equal_weight", "weight": POSITION_WEIGHT})


def build_specs() -> list[StrategySpec]:
    specs = []

    # --- 1-4: trend-following via SMA crossover ---
    specs.append(StrategySpec(
        name="trend_golden_cross_20_50",
        entry={"type": "sma_cross", "fast_window": 20, "slow_window": 50, "direction": "above"},
        exit={"type": "condition",
              "condition": {"type": "sma_cross", "fast_window": 20, "slow_window": 50, "direction": "below"},
              "max_days": 60},
        **COMMON,
    ))
    specs.append(StrategySpec(
        name="trend_death_cross_20_50_short",
        entry={"type": "sma_cross", "fast_window": 20, "slow_window": 50, "direction": "below"},
        exit={"type": "condition",
              "condition": {"type": "sma_cross", "fast_window": 20, "slow_window": 50, "direction": "above"},
              "max_days": 60},
        direction="short", **COMMON,
    ))
    specs.append(StrategySpec(
        name="trend_fast_cross_5_20",
        entry={"type": "sma_cross", "fast_window": 5, "slow_window": 20, "direction": "above"},
        exit={"type": "holding_period", "days": 5},
        **COMMON,
    ))
    specs.append(StrategySpec(
        name="trend_slow_golden_cross_50_200",
        entry={"type": "sma_cross", "fast_window": 50, "slow_window": 200, "direction": "above"},
        exit={"type": "condition",
              "condition": {"type": "sma_cross", "fast_window": 50, "slow_window": 200, "direction": "below"},
              "max_days": 90},
        **COMMON,
    ))

    # --- 5-8: RSI mean-reversion at extremes ---
    specs.append(StrategySpec(
        name="rsi_oversold_bounce_long",
        entry={"type": "rsi", "window": 14, "operator": "<", "value": 30},
        exit={"type": "condition", "condition": {"type": "rsi", "window": 14, "operator": ">", "value": 55}, "max_days": 10},
        **COMMON,
    ))
    specs.append(StrategySpec(
        name="rsi_overbought_fade_short",
        entry={"type": "rsi", "window": 14, "operator": ">", "value": 70},
        exit={"type": "condition", "condition": {"type": "rsi", "window": 14, "operator": "<", "value": 45}, "max_days": 10},
        direction="short", **COMMON,
    ))
    specs.append(StrategySpec(
        name="rsi_extreme_oversold_scalp_long",
        entry={"type": "rsi", "window": 14, "operator": "<", "value": 20},
        exit={"type": "stop_loss_take_profit", "stop_loss_pct": 0.03, "take_profit_pct": 0.06, "max_days": 5},
        **COMMON,
    ))
    specs.append(StrategySpec(
        name="rsi_extreme_overbought_scalp_short",
        entry={"type": "rsi", "window": 14, "operator": ">", "value": 80},
        exit={"type": "stop_loss_take_profit", "stop_loss_pct": 0.03, "take_profit_pct": 0.06, "max_days": 5},
        direction="short", **COMMON,
    ))

    # --- 9-10: regime filter (trade with the long-term trend) ---
    specs.append(StrategySpec(
        name="regime_above_sma200_long",
        entry={"type": "price_vs_sma", "window": 200, "operator": ">"},
        exit={"type": "holding_period", "days": 5},
        **COMMON,
    ))
    specs.append(StrategySpec(
        name="regime_below_sma200_short",
        entry={"type": "price_vs_sma", "window": 200, "operator": "<"},
        exit={"type": "holding_period", "days": 5},
        direction="short", **COMMON,
    ))

    # --- 11: momentum continuation, RSI-based exit instead of stop/take ---
    specs.append(StrategySpec(
        name="momentum_7pct_rsi_exit",
        entry={"type": "pct_change", "field": "close", "lookback_days": 1, "operator": ">", "value": 0.07},
        exit={"type": "condition", "condition": {"type": "rsi", "window": 14, "operator": ">", "value": 75}, "max_days": 5},
        **COMMON,
    ))

    # --- 12-15: the opposite bet — buy the dip / fade the spike ---
    specs.append(StrategySpec(
        name="buy_the_dip_3pct",
        entry={"type": "pct_change", "field": "close", "lookback_days": 1, "operator": "<", "value": -0.03},
        exit={"type": "holding_period", "days": 2},
        **COMMON,
    ))
    specs.append(StrategySpec(
        name="buy_the_dip_5pct_tight",
        entry={"type": "pct_change", "field": "close", "lookback_days": 1, "operator": "<", "value": -0.05},
        exit={"type": "stop_loss_take_profit", "stop_loss_pct": 0.02, "take_profit_pct": 0.04, "max_days": 3},
        **COMMON,
    ))
    specs.append(StrategySpec(
        name="fade_the_spike_5pct_short",
        entry={"type": "pct_change", "field": "close", "lookback_days": 1, "operator": ">", "value": 0.05},
        exit={"type": "holding_period", "days": 2},
        direction="short", **COMMON,
    ))
    specs.append(StrategySpec(
        name="fade_the_spike_7pct_tight_short",
        entry={"type": "pct_change", "field": "close", "lookback_days": 1, "operator": ">", "value": 0.07},
        exit={"type": "stop_loss_take_profit", "stop_loss_pct": 0.03, "take_profit_pct": 0.05, "max_days": 3},
        direction="short", **COMMON,
    ))

    # --- 16-17: short-horizon scalps off a reactive short SMA ---
    specs.append(StrategySpec(
        name="scalp_above_sma10_long",
        entry={"type": "price_vs_sma", "window": 10, "operator": ">"},
        exit={"type": "holding_period", "days": 1},
        **COMMON,
    ))
    specs.append(StrategySpec(
        name="scalp_below_sma10_short",
        entry={"type": "price_vs_sma", "window": 10, "operator": "<"},
        exit={"type": "holding_period", "days": 1},
        direction="short", **COMMON,
    ))

    # --- 18: trend-following, unbounded "let it run" exit ---
    specs.append(StrategySpec(
        name="trend_let_it_run_10_30",
        entry={"type": "sma_cross", "fast_window": 10, "slow_window": 30, "direction": "above"},
        exit={"type": "condition",
              "condition": {"type": "sma_cross", "fast_window": 10, "slow_window": 30, "direction": "below"}},
        **COMMON,
    ))

    # --- 19-20: mild (non-extreme) RSI mean-reversion ---
    specs.append(StrategySpec(
        name="rsi_mild_reversion_long",
        entry={"type": "rsi", "window": 14, "operator": "<", "value": 40},
        exit={"type": "holding_period", "days": 3},
        **COMMON,
    ))
    specs.append(StrategySpec(
        name="rsi_mild_reversion_short",
        entry={"type": "rsi", "window": 14, "operator": ">", "value": 60},
        exit={"type": "holding_period", "days": 3},
        direction="short", **COMMON,
    ))

    return specs


def main() -> None:
    print(f"Fetching {len(UNIVERSE)} tickers, {START} .. {END}...")
    data = fetch_universe(UNIVERSE, START, END)
    print("Data ready.\n")

    specs = build_specs()
    assert len(specs) == 20, f"expected 20 strategies, got {len(specs)}"

    rows = []
    for spec in specs:
        result = run_backtest(spec, data)
        metrics = compute_metrics(result, spec.initial_capital)
        log_run(spec.model_dump(mode="json"), metrics)
        rows.append({"name": spec.name, "direction": spec.direction, **metrics})
        ret = metrics["total_return_pct"] if metrics["total_return_pct"] is not None else 0
        sharpe = metrics["sharpe"] if metrics["sharpe"] is not None else 0
        wr = metrics["win_rate_pct"] if metrics["win_rate_pct"] is not None else 0
        print(f"{spec.name:36s} [{spec.direction:5s}] trades={metrics['num_trades']:4d}  "
              f"return={ret:7.2f}%  sharpe={sharpe:6.2f}  win_rate={wr:6.2f}%  "
              f"max_dd={metrics['max_drawdown_pct']:7.2f}%")

    print("\n=== Ranked by Sharpe (min 15 trades for significance) ===")
    ranked = sorted(
        (r for r in rows if r["num_trades"] >= 15 and r["sharpe"] is not None),
        key=lambda r: r["sharpe"], reverse=True,
    )
    for r in ranked[:10]:
        print(f"{r['name']:36s} [{r['direction']:5s}] trades={r['num_trades']:4d}  "
              f"return={r['total_return_pct']:7.2f}%  sharpe={r['sharpe']:6.2f}  "
              f"win_rate={r['win_rate_pct']:6.2f}%  cagr={r['cagr_pct']}%  max_dd={r['max_drawdown_pct']:7.2f}%")

    excluded = [r for r in rows if r["num_trades"] < 15]
    if excluded:
        print(f"\n({len(excluded)} strategies excluded from ranking, <15 trades — not statistically meaningful)")
        for r in excluded:
            print(f"  {r['name']}: {r['num_trades']} trades")


if __name__ == "__main__":
    main()
