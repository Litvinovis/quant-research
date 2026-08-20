import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engine import BacktestResult, Trade  # noqa: E402
from metrics import compute_metrics  # noqa: E402


def _trade(net_pnl_pct, gross_pnl_pct=None, ticker="A"):
    gross_pnl_pct = net_pnl_pct if gross_pnl_pct is None else gross_pnl_pct
    return Trade(
        ticker=ticker,
        entry_date=pd.Timestamp("2023-01-01"),
        entry_price=100.0,
        exit_date=pd.Timestamp("2023-01-02"),
        exit_price=100.0 * (1 + net_pnl_pct),
        exit_reason="holding_period",
        gross_pnl_pct=gross_pnl_pct,
        net_pnl_pct=net_pnl_pct,
    )


class TestEmptyResult:
    def test_no_trades_returns_zeroed_metrics(self):
        result = BacktestResult(
            spec_name="empty",
            trades=[],
            equity_curve=pd.Series([1_000_000.0], index=[pd.Timestamp("2023-01-01")]),
        )
        metrics = compute_metrics(result, initial_capital=1_000_000.0)
        assert metrics["num_trades"] == 0
        assert metrics["total_return_pct"] == 0.0
        assert metrics["total_return_pct_gross"] == 0.0
        assert metrics["cagr_pct"] is None
        assert metrics["cagr_pct_gross"] is None
        assert metrics["sharpe"] is None
        assert metrics["win_rate_pct"] is None


class TestBasicMetrics:
    @pytest.fixture
    def two_trade_result(self):
        curve = pd.Series(
            [1_000_000.0, 1_100_000.0, 990_000.0],
            index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-04"]),
        )
        gross_curve = pd.Series(
            [1_000_000.0, 1_100_000.0, 1_000_000.0],
            index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-04"]),
        )
        trades = [_trade(0.10, ticker="A"), _trade(-0.10, ticker="B")]
        return BacktestResult(spec_name="t", trades=trades, equity_curve=curve, gross_equity_curve=gross_curve)

    def test_total_return(self, two_trade_result):
        metrics = compute_metrics(two_trade_result, initial_capital=1_000_000.0)
        assert metrics["total_return_pct"] == pytest.approx(-1.0)

    def test_total_return_gross_uses_gross_equity_curve(self, two_trade_result):
        metrics = compute_metrics(two_trade_result, initial_capital=1_000_000.0)
        assert metrics["total_return_pct_gross"] == pytest.approx(0.0)

    def test_win_rate(self, two_trade_result):
        metrics = compute_metrics(two_trade_result, initial_capital=1_000_000.0)
        assert metrics["win_rate_pct"] == pytest.approx(50.0)

    def test_avg_trade_pnl_is_mean_of_net_trade_pnls(self, two_trade_result):
        metrics = compute_metrics(two_trade_result, initial_capital=1_000_000.0)
        assert metrics["avg_trade_pnl_pct"] == pytest.approx(0.0)

    def test_max_drawdown_measured_from_running_peak(self, two_trade_result):
        metrics = compute_metrics(two_trade_result, initial_capital=1_000_000.0)
        # peak was 1,100,000 after trade 1; dropped to 990,000 -> -10% from peak
        assert metrics["max_drawdown_pct"] == pytest.approx(-10.0)

    def test_num_trades_matches_trade_list_length(self, two_trade_result):
        metrics = compute_metrics(two_trade_result, initial_capital=1_000_000.0)
        assert metrics["num_trades"] == 2


class TestGrossVsNet:
    def test_gross_return_exceeds_net_when_commission_charged(self):
        # a single trade whose gross move is +5% but commission ate 2%
        curve = pd.Series([1_000_000.0, 1_030_000.0], index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
        gross_curve = pd.Series([1_000_000.0, 1_050_000.0], index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
        trades = [_trade(net_pnl_pct=0.03, gross_pnl_pct=0.05)]
        result = BacktestResult(spec_name="t", trades=trades, equity_curve=curve, gross_equity_curve=gross_curve)
        metrics = compute_metrics(result, initial_capital=1_000_000.0)
        assert metrics["total_return_pct_gross"] == pytest.approx(5.0)
        assert metrics["total_return_pct"] == pytest.approx(3.0)
        assert metrics["total_return_pct_gross"] > metrics["total_return_pct"]

    def test_cagr_gross_is_also_reported(self):
        curve = pd.Series([1_000_000.0, 1_030_000.0], index=pd.to_datetime(["2023-01-01", "2023-12-31"]))
        gross_curve = pd.Series([1_000_000.0, 1_050_000.0], index=pd.to_datetime(["2023-01-01", "2023-12-31"]))
        trades = [_trade(net_pnl_pct=0.03, gross_pnl_pct=0.05)]
        result = BacktestResult(spec_name="t", trades=trades, equity_curve=curve, gross_equity_curve=gross_curve)
        metrics = compute_metrics(result, initial_capital=1_000_000.0)
        assert metrics["cagr_pct_gross"] is not None
        assert metrics["cagr_pct_gross"] > metrics["cagr_pct"]


class TestCagr:
    def test_doubling_over_one_year_is_roughly_100pct(self):
        curve = pd.Series(
            [1_000_000.0, 2_000_000.0],
            index=pd.to_datetime(["2023-01-01", "2023-12-31"]),  # ~365 days later
        )
        result = BacktestResult(spec_name="t", trades=[_trade(1.0)], equity_curve=curve, gross_equity_curve=curve)
        metrics = compute_metrics(result, initial_capital=1_000_000.0)
        assert metrics["cagr_pct"] == pytest.approx(100.0, abs=2.0)

    def test_short_window_extreme_annualization_is_still_finite(self):
        # A 1-day 10% loss annualizes to something dramatic, not garbage/NaN —
        # documents the known limitation of CAGR on very short backtests.
        curve = pd.Series(
            [1_000_000.0, 900_000.0],
            index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
        )
        result = BacktestResult(spec_name="t", trades=[_trade(-0.10)], equity_curve=curve, gross_equity_curve=curve)
        metrics = compute_metrics(result, initial_capital=1_000_000.0)
        assert metrics["cagr_pct"] is not None
        assert metrics["cagr_pct"] < -50.0


class TestSharpe:
    def test_zero_variance_daily_returns_gives_none(self):
        # Equity only moves once then stays flat -> after the initial jump,
        # resampled daily returns are all 0 -> std()==0 -> sharpe is None.
        curve = pd.Series(
            [1_000_000.0] * 5,
            index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]),
        )
        result = BacktestResult(spec_name="t", trades=[_trade(0.0)], equity_curve=curve, gross_equity_curve=curve)
        metrics = compute_metrics(result, initial_capital=1_000_000.0)
        assert metrics["sharpe"] is None
