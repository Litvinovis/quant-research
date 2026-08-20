#!/usr/bin/env python3
"""
Takes the top-3 strategies by total_return_pct from the strategies
catalog and reruns each one, unchanged, on a "второй эшелон" (lower-tier,
less liquid) universe instead of the blue-chip one they were found on —
to check whether the edge is universe-specific or genuinely about the
entry/exit rule.

Pulls entry/exit/commission/position_sizing/date-range from the ORIGINAL
run's spec_json (via strategies.last_backtest_run_id) so the comparison
is apples-to-apples except for the universe swap.
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
TOP_N = 3

# Второй эшелон: liquid enough to have real data, clearly not blue-chip —
# deliberately non-overlapping with the 15-ticker universe the originals
# were found on (SBER/GAZP/LKOH/GMKN/NVTK/ROSN/TATN/MTSS/MGNT/CHMF/PLZL/
# ALRS/VTBR/RUAL/AFLT).
LOWER_TIER_UNIVERSE = [
    "AFKS", "MVID", "HYDR", "IRAO", "MSNG", "LSRG", "PIKK",
    "RTKM", "RASP", "BELU", "MTLR", "SGZH", "OZON",
]


def top_strategies_by_return(n: int) -> list[dict]:
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.name, s.total_return_pct, s.sharpe, br.spec_json
                FROM strategies s
                JOIN backtest_runs br ON br.id = s.last_backtest_run_id
                ORDER BY s.total_return_pct DESC
                LIMIT %s
                """,
                (n,),
            )
            return [
                {"name": name, "return": ret, "sharpe": sharpe, "spec_json": spec_json}
                for name, ret, sharpe, spec_json in cur.fetchall()
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
    top = top_strategies_by_return(TOP_N)
    print(f"Top {TOP_N} by return (original blue-chip universe):")
    for s in top:
        print(f"  {s['name']}: return={s['return']}% sharpe={s['sharpe']}")

    print(f"\nFetching lower-tier universe ({len(LOWER_TIER_UNIVERSE)} tickers)...")
    reference_spec = StrategySpec.model_validate(top[0]["spec_json"])
    data = fetch_universe_tolerant(LOWER_TIER_UNIVERSE, reference_spec.start_date, reference_spec.end_date)
    resolved = sorted(data.keys())
    print(f"Resolved {len(resolved)}/{len(LOWER_TIER_UNIVERSE)}: {resolved}\n")

    print("=== Results on lower-tier universe ===")
    for s in top:
        original = StrategySpec.model_validate(s["spec_json"])
        lower_tier_spec = original.model_copy(update={
            "name": f"{s['name']}_lowertier",
            "universe": resolved,
        })
        result = run_backtest(lower_tier_spec, data)
        metrics = compute_metrics(result, lower_tier_spec.initial_capital)
        log_run(lower_tier_spec.model_dump(mode="json"), metrics)

        print(f"{s['name']:24s} blue-chip: return={s['return']:7.2f}%  sharpe={s['sharpe']}")
        lt_return = metrics["total_return_pct"] if metrics["total_return_pct"] is not None else 0
        lt_sharpe = metrics["sharpe"] if metrics["sharpe"] is not None else 0
        print(f"{'':24s} lower-tier: return={lt_return:7.2f}%  sharpe={lt_sharpe:6.2f}  "
              f"trades={metrics['num_trades']}  win_rate={metrics['win_rate_pct']}%  "
              f"max_dd={metrics['max_drawdown_pct']}%")
        print()


if __name__ == "__main__":
    main()
