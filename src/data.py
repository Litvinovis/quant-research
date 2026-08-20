"""
Historical candle fetching from the Tinkoff Invest API, with a local
parquet cache so repeated backtests over the same ticker/range don't
re-hit the API.

TLS note: invest-public-api.tbank.ru serves a certificate chain rooted at
the Russian Ministry of Digital Development CA, which is not in most
systems' standard trust store. We pass that chain explicitly instead of
relying on grpc's default bundled roots (see certs/minsifry_chain.pem).
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import grpc
import pandas as pd
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.schemas import InstrumentIdType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data"
CERT_CHAIN_PATH = PROJECT_ROOT / "certs" / "minsifry_chain.pem"
DEFAULT_TARGET = os.environ.get("INVEST_API_TARGET_PY", "invest-public-api.tbank.ru:443")


def _make_channel(target: str) -> grpc.Channel:
    root_certs = CERT_CHAIN_PATH.read_bytes() if CERT_CHAIN_PATH.exists() else None
    creds = grpc.ssl_channel_credentials(root_certificates=root_certs)
    return grpc.secure_channel(target, creds)


def _client(token: str, target: str = DEFAULT_TARGET) -> Client:
    """Same as `Client(token, target=target)` but with our own channel so we
    can inject the Minцифры root CA chain."""
    client = Client(token, target=target)
    client._channel = _make_channel(target)  # noqa: SLF001 -- SDK gives no public hook for this
    return client


def _resolve_figi(services, ticker: str) -> str:
    result = services.instruments.find_instrument(query=ticker)
    for instrument in result.instruments:
        if instrument.ticker == ticker and instrument.class_code in ("TQBR", "SPBXM"):
            return instrument.figi
    for instrument in result.instruments:
        if instrument.ticker == ticker:
            return instrument.figi
    raise ValueError(f"Ticker not found via Tinkoff instruments search: {ticker}")


def _cache_path(ticker: str, start: date, end: date) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker}_{start.isoformat()}_{end.isoformat()}.parquet"


def fetch_candles(ticker: str, start: date, end: date, token: str | None = None) -> pd.DataFrame:
    """Daily OHLCV candles for one MOEX ticker, [start, end] inclusive.
    Cached to data/<ticker>_<start>_<end>.parquet after first fetch."""
    cache_file = _cache_path(ticker, start, end)
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    token = token or os.environ["TINKOFF_TOKEN"]
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc)

    with _client(token) as services:
        figi = _resolve_figi(services, ticker)
        rows = []
        for candle in services.get_all_candles(
            figi=figi,
            from_=start_dt,
            to=end_dt,
            interval=CandleInterval.CANDLE_INTERVAL_DAY,
        ):
            rows.append(
                {
                    "date": candle.time.date(),
                    "open": _quotation_to_float(candle.open),
                    "high": _quotation_to_float(candle.high),
                    "low": _quotation_to_float(candle.low),
                    "close": _quotation_to_float(candle.close),
                    "volume": candle.volume,
                }
            )

    if not rows:
        raise ValueError(f"No candles returned for {ticker} in [{start}, {end}]")

    df = pd.DataFrame(rows).set_index("date").sort_index()
    df.to_parquet(cache_file)
    return df


def _quotation_to_float(q) -> float:
    return q.units + q.nano / 1e9


def fetch_universe(tickers: list[str], start: date, end: date, token: str | None = None) -> dict[str, pd.DataFrame]:
    return {ticker: fetch_candles(ticker, start, end, token=token) for ticker in tickers}


def _intraday_cache_path(ticker: str, interval_name: str, start: date, end: date) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # separate filename shape from the daily cache (ticker_start_end.parquet)
    # specifically so the two can never collide on the same file.
    return CACHE_DIR / f"{ticker}_{interval_name}_{start.isoformat()}_{end.isoformat()}.parquet"


def fetch_intraday_candles(
    ticker: str, start: date, end: date, interval, token: str | None = None
) -> pd.DataFrame:
    """OHLCV candles at a sub-daily granularity (interval: a
    tinkoff.invest.CandleInterval like CANDLE_INTERVAL_5_MIN). Index is a
    tz-aware (UTC) Timestamp, not a date — multiple bars per day, unlike
    fetch_candles(). Cached separately from the daily cache."""
    interval_name = interval.name if hasattr(interval, "name") else str(interval)
    cache_file = _intraday_cache_path(ticker, interval_name, start, end)
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    token = token or os.environ["TINKOFF_TOKEN"]
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc)

    with _client(token) as services:
        figi = _resolve_figi(services, ticker)
        rows = []
        for candle in services.get_all_candles(
            figi=figi,
            from_=start_dt,
            to=end_dt,
            interval=interval,
        ):
            rows.append(
                {
                    "ts": candle.time,
                    "open": _quotation_to_float(candle.open),
                    "high": _quotation_to_float(candle.high),
                    "low": _quotation_to_float(candle.low),
                    "close": _quotation_to_float(candle.close),
                    "volume": candle.volume,
                }
            )

    if not rows:
        raise ValueError(f"No intraday candles returned for {ticker} in [{start}, {end}]")

    df = pd.DataFrame(rows).set_index("ts").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    df.to_parquet(cache_file)
    return df


def fetch_intraday_universe(
    tickers: list[str], start: date, end: date, interval, token: str | None = None
) -> dict[str, pd.DataFrame]:
    """Intraday candles are a lot more API calls than one daily fetch, so
    this paces requests and retries on rate limiting instead of just
    giving up on the first RESOURCE_EXHAUSTED."""
    data = {}
    for ticker in tickers:
        for attempt in range(3):
            try:
                data[ticker] = fetch_intraday_candles(ticker, start, end, interval, token=token)
                break
            except Exception as exc:  # noqa: BLE001 -- one bad/illiquid ticker must not kill the batch
                if "RESOURCE_EXHAUSTED" in str(exc) and attempt < 2:
                    wait = 10 * (attempt + 1)
                    print(f"[data] {ticker} rate-limited, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                print(f"[data] skipping {ticker}: {exc}")
                break
        time.sleep(0.5)  # pace requests to avoid tripping the limit in the first place
    return data
