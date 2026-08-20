# quant-research

A backtesting engine for MOEX trading strategies via the [Tinkoff Invest
API](https://tinkoff.github.io/investAPI/), built around one idea: a
strategy is a JSON spec from a small, fixed vocabulary of entry/exit
conditions — never free-form code. The engine that runs it is
deterministic and unit-tested, so the same spec on the same data always
produces the same number. This makes the tool a good fit for pairing with
an LLM: the model's job is to translate "buy when RSI drops below 30" into
the spec below, not to compute anything itself.

See [RESULTS.md](RESULTS.md) for a worked example — strategies tried,
what worked, what didn't, and two methodology traps that'll quietly wreck
a backtest number if you don't know to check for them.

## Quick start

```bash
./scripts/install.sh          # venv + deps (see note on tinkoff-investments below)
echo "TINKOFF_TOKEN=..." > .env   # a Tinkoff Invest API token (read-only is enough)

.venv/bin/python run.py --spec '{
  "name": "5pct_1day_momentum",
  "universe": ["SBER", "GAZP"],
  "start_date": "2023-01-01",
  "end_date": "2026-01-01",
  "entry": {"type": "pct_change", "field": "close", "lookback_days": 1, "operator": ">", "value": 0.05},
  "exit": {"type": "holding_period", "days": 1}
}'
```

## Strategy spec

- **universe**: MOEX tickers, e.g. `["SBER", "GAZP"]`
- **entry** — one condition:
  - `pct_change` — price move over N days (`field`, `lookback_days`, `operator`, `value` as a fraction: 0.05 = 5%)
  - `sma_cross` — fast SMA crosses slow SMA (`fast_window`, `slow_window`, `direction: above|below`)
  - `rsi` — RSI vs a threshold (`window`, `operator`, `value` 0-100)
  - `price_vs_sma` — price above/below its SMA (`window`, `operator`)
  - `consecutive_candles` — N bars in a row of one color (`count`, `candle_color: green|red`) — timeframe-agnostic, works on daily or intraday bars alike
- **exit** — one rule:
  - `holding_period` — hold N bars (not calendar days — works the same on 5-minute bars)
  - `stop_loss_take_profit` — `stop_loss_pct` / `take_profit_pct` / `max_days`
  - `condition` — same vocabulary as entry, plus a `max_days` safety cap
- **direction**: `long` (default) or `short` — flips P&L sign, stop/take-profit trigger direction, and slippage direction. Doesn't model borrow fees or short availability.
- **position_sizing**: `{"type": "equal_weight", "weight": 0.1}` — fraction of *current* equity per trade. Defaults to `1.0` (bet everything, every trade) — see the position-sizing trap in RESULTS.md before trusting a return number.
- **commission_pct**, **slippage_pct**, **initial_capital**

## Execution model

- A signal computed through bar T is acted on at bar T+1's open — no lookahead.
- `holding_period: N` closes at the close of bar `(entry + N − 1)`.
- `stop_loss_take_profit` / `condition` exits are checked bar-by-bar from entry; whichever triggers first wins, `max_days` caps it.
- One open position per ticker; trades across the universe are sequenced by entry date, each risking `weight` of equity at that point (compounding). Simultaneous overlapping positions across tickers aren't modeled.
- Every metric is reported both **net** (with `commission_pct`) and **gross** (same trades, no commission) — the gap tells you how much of an edge commission is eating.

## Data — `src/data.py`

Candles come from `tinkoff-investments`, cached to `data/*.parquet` so
repeat runs don't re-hit the API. Both daily (`fetch_candles`) and
intraday (`fetch_intraday_candles`, any `CandleInterval`) are supported,
with separate cache namespaces.

**TLS**: `invest-public-api.tbank.ru` serves a certificate chain rooted at
a CA (Russian Ministry of Digital Development) that's outside most
systems' default trust store. `certs/minsifry_chain.pem` is passed to the
grpc channel explicitly instead of relying on default bundled roots — if
you hit a TLS handshake failure against this API, this is almost
certainly why.

**Install note**: `tinkoff-investments`'s declared dependency on a
separate package named `tinkoff` doesn't resolve on PyPI as of writing.
`scripts/install.sh` installs it with `--no-deps` and pins the actual
transitive deps directly in `requirements.txt` — see the comment there.

## Optional: a rankable strategy catalog

`src/run_log.py` can log every backtest (spec + metrics, net and gross)
to Postgres — `db/schema.sql` defines `backtest_runs` (full history) and
`strategies` (one row per name, upserted with the latest result, rankable
by Sharpe). `grafana/backtest-runs-dashboard.json` is a starting-point
dashboard for it. None of this is required to use the engine —
`run.py --no-log` skips it entirely.

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

100+ tests, no network or database required — the broker client and
Postgres logging are mocked. Covers the DSL validation, every
condition/exit type (including direction and gross/net P&L), the no-
lookahead guarantee, and the data-caching logic.

## Known limitations

- Position sizing is sequential/compounding, not a true multi-asset
  portfolio simulation (see "Execution model" above).
- Short-selling doesn't model borrow cost or availability.
- No slippage/liquidity modeling beyond a flat `slippage_pct`.
- Backtested performance is not a promise of future performance,
  especially at the small sample sizes typical of a single strategy
  (see RESULTS.md for concrete trade counts).
