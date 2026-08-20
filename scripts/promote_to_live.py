#!/usr/bin/env python3
"""
Exports a strategies-catalog row into one JSON file per ticker in its
universe — a template for whatever live-execution system you plug this
into (one ticker per file keeps position tracking simple on that side).

Usage:
    .venv/bin/python scripts/promote_to_live.py momentum_7pct_sltp1
    .venv/bin/python scripts/promote_to_live.py momentum_7pct_sltp1 --mark-promoted
    .venv/bin/python scripts/promote_to_live.py momentum_7pct_sltp1 --ticker SBER  # just one ticker
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psycopg

DSN = "dbname=quant_research"
OUT_DIR = Path(__file__).resolve().parent.parent / "promoted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="strategies.name to promote")
    parser.add_argument("--ticker", help="export only this one ticker instead of the whole universe")
    parser.add_argument("--mark-promoted", action="store_true", help="set status='promoted_to_live' in the catalog")
    args = parser.parse_args()

    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT universe, entry, exit, direction, sharpe, total_return_pct, num_trades, status "
                "FROM strategies WHERE name = %s",
                (args.name,),
            )
            row = cur.fetchone()
            if row is None:
                print(f"No strategy named {args.name!r} in the catalog.", file=sys.stderr)
                sys.exit(1)
            universe, entry, exit_rule, direction, sharpe, total_return, num_trades, status = row

            tickers = [args.ticker] if args.ticker else universe
            if args.ticker and args.ticker not in universe:
                print(f"Warning: {args.ticker!r} is not in {args.name}'s backtested universe {universe}", file=sys.stderr)

            print(f"{args.name}: direction={direction} sharpe={sharpe} return={total_return}% trades={num_trades} status={status}")
            OUT_DIR.mkdir(exist_ok=True)
            for ticker in tickers:
                config = {"name": args.name, "ticker": ticker, "direction": direction, "entry": entry, "exit": exit_rule}
                out_path = OUT_DIR / f"{args.name}_{ticker}.json"
                out_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
                print(f"  wrote {out_path}")

            if args.mark_promoted:
                cur.execute("UPDATE strategies SET status = 'promoted_to_live', updated_at = now() WHERE name = %s", (args.name,))
                print("  status -> promoted_to_live")

    print("\nNothing trades on its own — these are just config files. Wire them into")
    print("your own execution system (which needs to handle real-money risk controls")
    print("this repo doesn't: position limits, circuit breakers, a kill switch, a")
    print("sandbox/paper-trading mode before anything real).")


if __name__ == "__main__":
    main()
