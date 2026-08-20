-- quant_research database schema.
--
-- One-time setup (adjust role/auth to your own Postgres setup — this
-- assumes a local peer-auth role matching your OS user, no password,
-- via `local all all peer` in pg_hba.conf):
--
--   sudo -u postgres psql -c "CREATE ROLE your_user LOGIN;"
--   sudo -u postgres psql -c "CREATE DATABASE quant_research OWNER your_user;"
--   psql -d quant_research -f db/schema.sql
--
-- src/run_log.py writes one row here per run.py invocation (best-effort —
-- a DB error never fails the backtest itself). Grafana reads this table
-- directly via its Postgres datasource for the run-history dashboard.

CREATE TABLE IF NOT EXISTS backtest_runs (
    id BIGSERIAL PRIMARY KEY,
    run_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    spec_name TEXT NOT NULL,
    universe TEXT[] NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    spec_json JSONB NOT NULL,
    num_trades INT NOT NULL,
    total_return_pct DOUBLE PRECISION,
    total_return_pct_gross DOUBLE PRECISION,  -- same trades, commission_pct=0 — shows how much commission ate
    cagr_pct DOUBLE PRECISION,
    cagr_pct_gross DOUBLE PRECISION,
    max_drawdown_pct DOUBLE PRECISION,
    sharpe DOUBLE PRECISION,
    win_rate_pct DOUBLE PRECISION,
    avg_trade_pnl_pct DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_run_at ON backtest_runs (run_at);

-- Curated, rankable catalog of named strategies — one row per spec `name`,
-- upserted by src/run_log.py::register_strategy every time that name is
-- backtested. backtest_runs above stays the full raw history (every run,
-- every sweep variant); this table is "what's the latest known
-- performance of strategy X", the thing you actually rank and pick from
-- to export for live use (see scripts/promote_to_live.py).
CREATE TABLE IF NOT EXISTS strategies (
    id BIGSERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    universe TEXT[] NOT NULL,
    entry JSONB NOT NULL,
    exit JSONB NOT NULL,
    direction TEXT NOT NULL DEFAULT 'long' CHECK (direction IN ('long', 'short')),
    commission_pct DOUBLE PRECISION NOT NULL,
    position_weight DOUBLE PRECISION NOT NULL,
    last_backtest_run_id BIGINT REFERENCES backtest_runs(id),
    num_trades INT,
    total_return_pct DOUBLE PRECISION,
    total_return_pct_gross DOUBLE PRECISION,
    cagr_pct DOUBLE PRECISION,
    cagr_pct_gross DOUBLE PRECISION,
    sharpe DOUBLE PRECISION,
    win_rate_pct DOUBLE PRECISION,
    max_drawdown_pct DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate', 'promoted_to_live', 'retired')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_strategies_sharpe ON strategies (sharpe DESC NULLS LAST);
