#!/usr/bin/env python3
"""
Reruns every trend-following strategy (SMA crossover based, name prefix
"trend_") from the catalog, unchanged, on the same lower-tier
("второй эшелон") universe as lower_tier_test.py — to check whether
yesterday's stray finding (trend_death_cross_20_50_short did BETTER on
lower-tier than blue-chip) holds for trend-following as a category, or
was a one-off.

Unlike lower_tier_test.py (which picks "top 3 by return" from whatever's
currently in the catalog), this pulls ALL trend_* strategies by name, so
the category gets a full, deliberate test rather than whichever ones
happened to rank highest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from data import fetch_candles  # noqa: E402
from engine import run_backtest  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from run_log import log_run  # noqa: E402
from strategy_spec import StrategySpec  # noqa: E402

DSN = "dbname=quant_research"

# Same second-echelon universe as lower_tier_test.py, for direct comparison.
LOWER_TIER_UNIVERSE = [
    "AFKS", "MVID", "HYDR", "IRAO", "MSNG", "LSRG", "PIKK",
    "RTKM", "RASP", "BELU", "MTLR", "SGZH", "OZON",
]


def trend_strategies() -> list[dict]:
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.name, s.total_return_pct, s.sharpe, s.num_trades, br.spec_json
                FROM strategies s
                JOIN backtest_runs br ON br.id = s.last_backtest_run_id
                WHERE s.name LIKE 'trend_%' AND s.name NOT LIKE '%lowertier%'
                ORDER BY s.name
                """,
            )
            return [
                {"name": name, "return": ret, "sharpe": sharpe, "trades": trades, "spec_json": spec_json}
                for name, ret, sharpe, trades, spec_json in cur.fetchall()
            ]


def fetch_universe_tolerant(tickers: list[str], start, end) -> dict:
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = fetch_candles(ticker, start, end)
        except Exception as exc:  # noqa: BLE001 -- one bad ticker must not kill the batch
            print(f"  skipping {ticker}: {exc}")
    return data


def main() -> None:
    strategies = trend_strategies()
    if not strategies:
        print("No trend_* strategies found in the catalog.", file=sys.stderr)
        sys.exit(1)

    print(f"Trend-following strategies to retest ({len(strategies)}):")
    for s in strategies:
        print(f"  {s['name']}: blue-chip return={s['return']}% sharpe={s['sharpe']} trades={s['trades']}")

    reference_spec = StrategySpec.model_validate(strategies[0]["spec_json"])
    print(f"\nFetching lower-tier universe ({len(LOWER_TIER_UNIVERSE)} tickers), "
          f"{reference_spec.start_date} .. {reference_spec.end_date}...")
    data = fetch_universe_tolerant(LOWER_TIER_UNIVERSE, reference_spec.start_date, reference_spec.end_date)
    resolved = sorted(data.keys())
    print(f"Resolved {len(resolved)}/{len(LOWER_TIER_UNIVERSE)}: {resolved}\n")

    print("=== Results: blue-chip vs lower-tier ===")
    for s in strategies:
        original = StrategySpec.model_validate(s["spec_json"])
        lower_tier_spec = original.model_copy(update={
            "name": f"{s['name']}_lowertier2",
            "universe": resolved,
        })
        result = run_backtest(lower_tier_spec, data)
        metrics = compute_metrics(result, lower_tier_spec.initial_capital)
        log_run(lower_tier_spec.model_dump(mode="json"), metrics)

        lt_return = metrics["total_return_pct"] if metrics["total_return_pct"] is not None else 0
        lt_sharpe = metrics["sharpe"] if metrics["sharpe"] is not None else 0
        print(f"{s['name']:32s} blue-chip:  return={s['return']:7.2f}%  sharpe={s['sharpe']}  trades={s['trades']}")
        print(f"{'':32s} lower-tier: return={lt_return:7.2f}%  sharpe={lt_sharpe:6.2f}  "
              f"trades={metrics['num_trades']}  win_rate={metrics['win_rate_pct']}%  "
              f"max_dd={metrics['max_drawdown_pct']}%")
        print()


if __name__ == "__main__":
    main()
