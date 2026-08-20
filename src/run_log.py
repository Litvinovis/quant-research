"""
Best-effort logging of backtest runs to Postgres, so Grafana can chart run
history. Connects via local peer auth (no password) to db `quant_research`.

Never allowed to break a backtest: any DB error is caught and reported to
stderr, the caller still gets its result.
"""
from __future__ import annotations

import json
import sys

import psycopg

DSN = "dbname=quant_research"


def log_run(spec: dict, metrics: dict) -> None:
    """Logs the run to backtest_runs (full history) and upserts the
    strategies catalog (latest-known-performance-by-name) in the same
    connection. Both best-effort — a DB error here must never fail the
    backtest itself."""
    try:
        with psycopg.connect(DSN, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO backtest_runs
                        (spec_name, universe, start_date, end_date, spec_json,
                         num_trades, total_return_pct, total_return_pct_gross,
                         cagr_pct, cagr_pct_gross, max_drawdown_pct,
                         sharpe, win_rate_pct, avg_trade_pnl_pct)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        spec["name"],
                        spec["universe"],
                        spec["start_date"],
                        spec["end_date"],
                        json.dumps(spec, ensure_ascii=False),
                        metrics["num_trades"],
                        metrics["total_return_pct"],
                        metrics["total_return_pct_gross"],
                        metrics["cagr_pct"],
                        metrics["cagr_pct_gross"],
                        metrics["max_drawdown_pct"],
                        metrics["sharpe"],
                        metrics["win_rate_pct"],
                        metrics["avg_trade_pnl_pct"],
                    ),
                )
                (run_id,) = cur.fetchone()
                _upsert_strategy(cur, spec, metrics, run_id)
    except Exception as exc:  # noqa: BLE001 -- logging is best-effort, never fatal
        print(f"[run_log] failed to log run to Postgres: {exc}", file=sys.stderr)


def _upsert_strategy(cur, spec: dict, metrics: dict, run_id: int) -> None:
    """Every named spec that gets backtested is a catalog candidate —
    ranking (min trade count etc.) is a query-time filter, not a gate on
    what gets registered here."""
    cur.execute(
        """
        INSERT INTO strategies
            (name, universe, entry, exit, direction, commission_pct, position_weight,
             last_backtest_run_id, num_trades, total_return_pct, total_return_pct_gross,
             cagr_pct, cagr_pct_gross, sharpe, win_rate_pct, max_drawdown_pct, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (name) DO UPDATE SET
            universe = EXCLUDED.universe,
            entry = EXCLUDED.entry,
            exit = EXCLUDED.exit,
            direction = EXCLUDED.direction,
            commission_pct = EXCLUDED.commission_pct,
            position_weight = EXCLUDED.position_weight,
            last_backtest_run_id = EXCLUDED.last_backtest_run_id,
            num_trades = EXCLUDED.num_trades,
            total_return_pct = EXCLUDED.total_return_pct,
            total_return_pct_gross = EXCLUDED.total_return_pct_gross,
            cagr_pct = EXCLUDED.cagr_pct,
            cagr_pct_gross = EXCLUDED.cagr_pct_gross,
            sharpe = EXCLUDED.sharpe,
            win_rate_pct = EXCLUDED.win_rate_pct,
            max_drawdown_pct = EXCLUDED.max_drawdown_pct,
            updated_at = now()
        """,
        (
            spec["name"],
            spec["universe"],
            json.dumps(spec["entry"], ensure_ascii=False),
            json.dumps(spec["exit"], ensure_ascii=False),
            spec.get("direction", "long"),
            spec["commission_pct"],
            spec["position_sizing"]["weight"],
            run_id,
            metrics["num_trades"],
            metrics["total_return_pct"],
            metrics["total_return_pct_gross"],
            metrics["cagr_pct"],
            metrics["cagr_pct_gross"],
            metrics["sharpe"],
            metrics["win_rate_pct"],
            metrics["max_drawdown_pct"],
        ),
    )
