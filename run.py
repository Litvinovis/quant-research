#!/usr/bin/env python3
"""
Entry point for running one backtest from a StrategySpec JSON.

Usage:
    .venv/bin/python run.py --spec '{"name": "...", "universe": ["SBER"], ...}'
    .venv/bin/python run.py --spec-file spec.json
    echo '{...}' | .venv/bin/python run.py

Prints a single JSON object to stdout: {"spec": ..., "metrics": ..., "trades": [...]}
Never prints TINKOFF_TOKEN or any credential.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent / ".env")

from data import fetch_universe  # noqa: E402
from engine import run_backtest  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from run_log import log_run  # noqa: E402
from strategy_spec import StrategySpec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", help="StrategySpec as a JSON string")
    parser.add_argument("--spec-file", help="Path to a JSON file with the StrategySpec")
    parser.add_argument(
        "--no-log", action="store_true",
        help="Skip writing to Postgres (backtest_runs/strategies) — for pipeline "
             "smoke tests and other runs that aren't real research and shouldn't "
             "need manual DB cleanup afterward.",
    )
    args = parser.parse_args()

    if args.spec:
        raw = args.spec
    elif args.spec_file:
        raw = Path(args.spec_file).read_text()
    else:
        raw = sys.stdin.read()

    spec = StrategySpec.model_validate_json(raw)
    data = fetch_universe(spec.universe, spec.start_date, spec.end_date)
    result = run_backtest(spec, data)
    metrics = compute_metrics(result, spec.initial_capital)
    spec_dict = spec.model_dump(mode="json")
    if not args.no_log:
        log_run(spec_dict, metrics)

    output = {
        "spec": spec_dict,
        "metrics": metrics,
        "trades": [
            {
                "ticker": t.ticker,
                "entry_date": str(t.entry_date),
                "entry_price": round(t.entry_price, 4),
                "exit_date": str(t.exit_date),
                "exit_price": round(t.exit_price, 4),
                "exit_reason": t.exit_reason,
                "gross_pnl_pct": round(t.gross_pnl_pct * 100, 3),
                "net_pnl_pct": round(t.net_pnl_pct * 100, 3),
            }
            for t in result.trades
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
