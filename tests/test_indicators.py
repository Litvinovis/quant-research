import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indicators import pct_change, rsi, sma  # noqa: E402


class TestPctChange:
    def test_one_day_lookback(self):
        s = pd.Series([100.0, 110.0, 99.0, 99.0])
        result = pct_change(s, 1)
        assert result.iloc[1] == pytest.approx(0.10)
        assert result.iloc[2] == pytest.approx((99 - 110) / 110)
        assert result.iloc[3] == pytest.approx(0.0)

    def test_first_value_is_nan(self):
        s = pd.Series([100.0, 110.0])
        assert pd.isna(pct_change(s, 1).iloc[0])

    def test_multi_day_lookback(self):
        s = pd.Series([100.0, 105.0, 110.0, 121.0])
        result = pct_change(s, 2)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(0.10)  # 100 -> 110
        assert result.iloc[3] == pytest.approx(0.15238, abs=1e-4)  # 105 -> 121


class TestSma:
    def test_basic_average(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)  # mean(1,2,3)
        assert result.iloc[3] == pytest.approx(3.0)  # mean(2,3,4)
        assert result.iloc[4] == pytest.approx(4.0)  # mean(3,4,5)

    def test_window_one_equals_series(self):
        s = pd.Series([5.0, 7.0, 2.0])
        result = sma(s, 1)
        pd.testing.assert_series_equal(result, s, check_names=False)


class TestRsi:
    def test_all_gains_saturates_at_100(self):
        s = pd.Series([100.0 + i for i in range(20)])  # strictly increasing
        result = rsi(s, window=5)
        assert result.iloc[-1] == pytest.approx(100.0)

    def test_all_losses_saturates_at_0(self):
        s = pd.Series([100.0 - i for i in range(20)])  # strictly decreasing
        result = rsi(s, window=5)
        assert result.iloc[-1] == pytest.approx(0.0)

    def test_flat_series_is_50(self):
        s = pd.Series([100.0] * 20)  # no gains, no losses
        result = rsi(s, window=5)
        assert result.iloc[-1] == pytest.approx(50.0)

    def test_bounded_between_0_and_100(self):
        rng = np.random.default_rng(42)
        s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 100)))
        result = rsi(s, window=14).dropna()
        assert (result >= 0).all()
        assert (result <= 100).all()

    def test_no_nan_after_window_even_with_zero_losses(self):
        # Regression: avg_loss==0 must not produce NaN (see indicators.rsi docstring)
        s = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        result = rsi(s, window=3)
        assert not result.iloc[3:].isna().any()
