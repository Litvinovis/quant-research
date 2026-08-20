import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import data  # noqa: E402
from data import (  # noqa: E402
    _quotation_to_float,
    _resolve_figi,
    fetch_candles,
    fetch_intraday_candles,
    fetch_intraday_universe,
)

FAKE_5MIN_INTERVAL = SimpleNamespace(name="CANDLE_INTERVAL_5_MIN")


@dataclass
class FakeQuotation:
    units: int
    nano: int


class TestQuotationToFloat:
    def test_positive_price(self):
        assert _quotation_to_float(FakeQuotation(units=100, nano=500_000_000)) == pytest.approx(100.5)

    def test_zero_nano(self):
        assert _quotation_to_float(FakeQuotation(units=250, nano=0)) == pytest.approx(250.0)

    def test_small_fractional_part(self):
        assert _quotation_to_float(FakeQuotation(units=1, nano=10_000_000)) == pytest.approx(1.01)


class TestResolveFigi:
    def _services(self, instruments):
        find_instrument_result = SimpleNamespace(instruments=instruments)
        return SimpleNamespace(instruments=SimpleNamespace(find_instrument=lambda query: find_instrument_result))

    def test_prefers_moex_class_code_over_other_matches(self):
        instruments = [
            SimpleNamespace(ticker="SBER", class_code="SPBAM", figi="wrong-figi"),
            SimpleNamespace(ticker="SBER", class_code="TQBR", figi="moex-figi"),
        ]
        figi = _resolve_figi(self._services(instruments), "SBER")
        assert figi == "moex-figi"

    def test_falls_back_to_any_ticker_match_if_no_preferred_class_code(self):
        instruments = [SimpleNamespace(ticker="SBER", class_code="SPBAM", figi="fallback-figi")]
        figi = _resolve_figi(self._services(instruments), "SBER")
        assert figi == "fallback-figi"

    def test_raises_when_ticker_not_found(self):
        with pytest.raises(ValueError, match="Ticker not found"):
            _resolve_figi(self._services([]), "NOTATICKER")


class TestFetchCandlesCache:
    def test_cache_hit_returns_data_without_needing_token_or_network(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
        monkeypatch.delenv("TINKOFF_TOKEN", raising=False)

        start, end = date(2024, 1, 1), date(2024, 1, 5)
        cache_file = tmp_path / f"SBER_{start.isoformat()}_{end.isoformat()}.parquet"
        expected = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000]},
            index=pd.Index([date(2024, 1, 2)], name="date"),
        )
        expected.to_parquet(cache_file)

        result = fetch_candles("SBER", start, end)  # no token passed, none in env
        pd.testing.assert_frame_equal(result, expected)

    def test_cache_miss_without_token_raises_instead_of_silently_failing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
        monkeypatch.delenv("TINKOFF_TOKEN", raising=False)

        with pytest.raises(KeyError):
            fetch_candles("SBER", date(2024, 1, 1), date(2024, 1, 5))


class TestFetchIntradayCandlesCache:
    def test_cache_file_name_includes_interval_so_it_cannot_collide_with_daily(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
        monkeypatch.delenv("TINKOFF_TOKEN", raising=False)

        start, end = date(2024, 1, 1), date(2024, 1, 5)
        cache_file = tmp_path / f"SBER_CANDLE_INTERVAL_5_MIN_{start.isoformat()}_{end.isoformat()}.parquet"
        expected = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000]},
            index=pd.DatetimeIndex([datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)], name="ts"),
        )
        expected.to_parquet(cache_file)

        result = fetch_intraday_candles("SBER", start, end, FAKE_5MIN_INTERVAL)
        pd.testing.assert_frame_equal(result, expected)

    def test_cache_miss_without_token_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
        monkeypatch.delenv("TINKOFF_TOKEN", raising=False)

        with pytest.raises(KeyError):
            fetch_intraday_candles("SBER", date(2024, 1, 1), date(2024, 1, 5), FAKE_5MIN_INTERVAL)


class TestFetchIntradayUniverse:
    def test_one_bad_ticker_does_not_kill_the_batch(self, tmp_path, monkeypatch):
        monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
        monkeypatch.delenv("TINKOFF_TOKEN", raising=False)
        monkeypatch.setattr(data.time, "sleep", lambda *_: None)  # skip the real pacing delay in tests

        start, end = date(2024, 1, 1), date(2024, 1, 5)
        good_cache = tmp_path / f"SBER_CANDLE_INTERVAL_5_MIN_{start.isoformat()}_{end.isoformat()}.parquet"
        pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000]},
            index=pd.DatetimeIndex([datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)], name="ts"),
        ).to_parquet(good_cache)
        # "BADTICKER" has no cache file and no token -> fetch raises internally, must be skipped

        result = fetch_intraday_universe(["SBER", "BADTICKER"], start, end, FAKE_5MIN_INTERVAL)
        assert list(result.keys()) == ["SBER"]
