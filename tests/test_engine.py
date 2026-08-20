import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engine import _filter_by_date_range, _find_exit, evaluate_condition, run_backtest  # noqa: E402
from strategy_spec import (  # noqa: E402
    ConditionExit,
    ConsecutiveCandlesCondition,
    HoldingPeriodExit,
    PctChangeCondition,
    PositionSizing,
    PriceVsSmaCondition,
    RsiCondition,
    SmaCrossCondition,
    StopLossTakeProfitExit,
    StrategySpec,
)


def _dates(n, start=date(2024, 1, 1)):
    return [start + timedelta(days=i) for i in range(n)]


def _ohlc_df(closes, opens=None, dates=None):
    opens = opens or closes
    dates = dates or _dates(len(closes))
    return pd.DataFrame(
        {"open": opens, "high": closes, "low": opens, "close": closes, "volume": [1000] * len(closes)},
        index=dates,
    )


@pytest.fixture
def synthetic_df():
    # plain date objects, matching data.py's real fetch_candles() index dtype
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    return pd.DataFrame(
        {
            "open": [100, 101, 107, 108, 109],
            "high": [100, 106, 107, 108, 109],
            "low": [100, 101, 105, 108, 109],
            "close": [100, 106, 105, 108, 109],  # day1: +6% vs day0 -> signal
            "volume": [1000] * 5,
        },
        index=dates,
    )


def test_pct_change_entry_and_holding_period_exit(synthetic_df):
    spec = StrategySpec(
        name="test",
        universe=["TEST"],
        start_date="2024-01-01",
        end_date="2024-01-05",
        entry=PctChangeCondition(field="close", lookback_days=1, operator=">", value=0.05),
        exit=HoldingPeriodExit(days=1),
        commission_pct=0.0,
        slippage_pct=0.0,
    )

    result = run_backtest(spec, {"TEST": synthetic_df})

    assert len(result.trades) == 1
    trade = result.trades[0]
    # signal fires on day1 (close 106, +6% vs day0's 100) -> entry at day2's open (107)
    assert trade.entry_date == date(2024, 1, 3)
    assert trade.entry_price == pytest.approx(107.0)
    # holding_period days=1 -> exit at close of the entry bar itself
    assert trade.exit_date == date(2024, 1, 3)
    assert trade.exit_price == pytest.approx(105.0)
    assert trade.net_pnl_pct == pytest.approx((105.0 - 107.0) / 107.0)
    assert trade.gross_pnl_pct == pytest.approx(trade.net_pnl_pct)  # commission_pct=0 here


def test_no_signal_means_no_trades(synthetic_df):
    spec = StrategySpec(
        name="test",
        universe=["TEST"],
        start_date="2024-01-01",
        end_date="2024-01-05",
        entry=PctChangeCondition(field="close", lookback_days=1, operator=">", value=0.50),
        exit=HoldingPeriodExit(days=1),
    )
    result = run_backtest(spec, {"TEST": synthetic_df})
    assert len(result.trades) == 0
    assert result.equity_curve.iloc[-1] == spec.initial_capital


class TestFilterByDateRange:
    def test_date_indexed_daily_bars(self):
        # object-dtype Index of plain `date` objects — data.py::fetch_candles' shape
        idx = pd.Index([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 5)])
        df = pd.DataFrame({"close": [1, 2, 3]}, index=idx)
        result = _filter_by_date_range(df, date(2024, 1, 2), date(2024, 1, 2))
        assert list(result.index) == [date(2024, 1, 2)]

    def test_tz_aware_intraday_bars(self):
        # tz-aware DatetimeIndex — intraday candle fetchers' shape. This is
        # the regression: comparing this to a plain `date` bound used to
        # raise "Invalid comparison between dtype=datetime64[...] and date".
        idx = pd.to_datetime([
            "2024-01-01 10:00", "2024-01-01 10:05", "2024-01-02 10:00",
        ]).tz_localize("UTC")
        df = pd.DataFrame({"close": [1, 2, 3]}, index=idx)
        result = _filter_by_date_range(df, date(2024, 1, 1), date(2024, 1, 1))
        assert len(result) == 2  # both Jan-1 bars, none of Jan-2

    def test_intraday_end_date_is_inclusive_of_the_whole_day(self):
        idx = pd.to_datetime(["2024-01-01 23:55", "2024-01-02 00:00"]).tz_localize("UTC")
        df = pd.DataFrame({"close": [1, 2]}, index=idx)
        result = _filter_by_date_range(df, date(2024, 1, 1), date(2024, 1, 1))
        assert len(result) == 1  # the 23:55 bar counts as "on" Jan-1; the next-day 00:00 bar doesn't


class TestNoLookahead:
    def test_signal_on_last_bar_produces_no_trade(self, synthetic_df):
        # Rig the entry condition to fire only on the very last bar (no
        # next bar to execute on) — the engine must not fabricate a trade.
        df = synthetic_df.copy()
        df.loc[df.index[-1], "close"] = df["open"].iloc[-1] * 2  # huge jump on last bar
        spec = StrategySpec(
            name="test",
            universe=["TEST"],
            start_date="2024-01-01",
            end_date="2024-01-05",
            entry=PctChangeCondition(field="close", lookback_days=1, operator=">", value=0.5),
            exit=HoldingPeriodExit(days=1),
        )
        result = run_backtest(spec, {"TEST": df})
        assert len(result.trades) == 0


class TestCommissionAndSlippage:
    def test_commission_reduces_net_pnl_by_round_trip_amount(self, synthetic_df):
        spec_with_cost = StrategySpec(
            name="b", universe=["TEST"], start_date="2024-01-01", end_date="2024-01-05",
            entry=PctChangeCondition(field="close", lookback_days=1, operator=">", value=0.05),
            exit=HoldingPeriodExit(days=1), commission_pct=0.01, slippage_pct=0.0,
        )
        trade = run_backtest(spec_with_cost, {"TEST": synthetic_df}).trades[0]
        # commission is charged on both entry and exit legs; gross itself
        # is unaffected by commission_pct (same price move either way)
        assert trade.net_pnl_pct == pytest.approx(trade.gross_pnl_pct - 2 * 0.01)

    def test_zero_commission_makes_gross_and_net_equal(self, synthetic_df):
        spec = StrategySpec(
            name="a", universe=["TEST"], start_date="2024-01-01", end_date="2024-01-05",
            entry=PctChangeCondition(field="close", lookback_days=1, operator=">", value=0.05),
            exit=HoldingPeriodExit(days=1), commission_pct=0.0, slippage_pct=0.0,
        )
        trade = run_backtest(spec, {"TEST": synthetic_df}).trades[0]
        assert trade.gross_pnl_pct == pytest.approx(trade.net_pnl_pct)

    def test_gross_equity_curve_ignores_commission(self, synthetic_df):
        spec = StrategySpec(
            name="c", universe=["TEST"], start_date="2024-01-01", end_date="2024-01-05",
            entry=PctChangeCondition(field="close", lookback_days=1, operator=">", value=0.05),
            exit=HoldingPeriodExit(days=1), commission_pct=0.02, slippage_pct=0.0,
        )
        result = run_backtest(spec, {"TEST": synthetic_df})
        # net curve is worse than gross curve once commission is nonzero
        assert result.gross_equity_curve.iloc[-1] > result.equity_curve.iloc[-1]

    def test_slippage_worsens_entry_and_exit_price(self, synthetic_df):
        spec = StrategySpec(
            name="test", universe=["TEST"], start_date="2024-01-01", end_date="2024-01-05",
            entry=PctChangeCondition(field="close", lookback_days=1, operator=">", value=0.05),
            exit=HoldingPeriodExit(days=1), commission_pct=0.0, slippage_pct=0.02,
        )
        trade = run_backtest(spec, {"TEST": synthetic_df}).trades[0]
        # entry at open (107) paid worse by +2%; exit at close (105) received worse by -2%
        assert trade.entry_price == pytest.approx(107.0 * 1.02)
        assert trade.exit_price == pytest.approx(105.0 * 0.98)


class TestStopLossTakeProfitExit:
    @pytest.fixture
    def drop_then_spike_df(self):
        # entry bar has close==open (no move); then -2%, -5% (stop hit), +15% (would be TP, too late)
        closes = [100, 100, 98, 95, 115]
        return _ohlc_df(closes)

    def test_stop_loss_fires_before_later_take_profit(self, drop_then_spike_df):
        exit_rule = StopLossTakeProfitExit(stop_loss_pct=0.03, take_profit_pct=0.08, max_days=10)
        exit_i, exit_price, reason = _find_exit(exit_rule, drop_then_spike_df, entry_i=0)
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(95.0)
        assert drop_then_spike_df.index[exit_i] == drop_then_spike_df.index[3]

    def test_take_profit_fires_when_hit_first(self):
        closes = [100, 100, 112, 90]  # +12% on bar 2, before any stop-loss level
        df = _ohlc_df(closes)
        exit_rule = StopLossTakeProfitExit(stop_loss_pct=0.20, take_profit_pct=0.08, max_days=10)
        exit_i, exit_price, reason = _find_exit(exit_rule, df, entry_i=0)
        assert reason == "take_profit"
        assert exit_price == pytest.approx(112.0)

    def test_max_days_fallback_when_neither_bound_hit(self):
        closes = [100, 100, 101, 99, 100]  # stays within +-3% of entry the whole time
        df = _ohlc_df(closes)
        exit_rule = StopLossTakeProfitExit(stop_loss_pct=0.10, take_profit_pct=0.10, max_days=3)
        exit_i, exit_price, reason = _find_exit(exit_rule, df, entry_i=0)
        assert reason == "max_days"
        assert exit_i == 3  # entry_i(0) + max_days(3)

    def test_at_least_one_bound_required_by_schema(self):
        with pytest.raises(Exception):
            StopLossTakeProfitExit()


class TestShortDirection:
    def test_pnl_is_positive_when_price_falls_after_short_entry(self):
        # bad-news drop signal on bar 1 (close -6%) -> short entry at bar 2's
        # open; price keeps falling through the holding period -> profit.
        # day-over-day changes after bar 1 stay well above -5% so this is
        # the only signal (no second entry once the first trade closes).
        closes = [100, 94, 93, 92, 91]
        opens = [100, 99, 95, 92, 91]
        df = _ohlc_df(closes, opens=opens)
        spec = StrategySpec(
            name="short_test", universe=["TEST"], start_date="2024-01-01", end_date="2024-01-05",
            entry=PctChangeCondition(field="close", lookback_days=1, operator="<", value=-0.05),
            exit=HoldingPeriodExit(days=1), direction="short",
            commission_pct=0.0, slippage_pct=0.0,
        )
        result = run_backtest(spec, {"TEST": df})
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.entry_price == pytest.approx(95.0)  # open of entry bar
        assert trade.exit_price == pytest.approx(93.0)  # close of same bar (days=1)
        assert trade.net_pnl_pct == pytest.approx((95.0 - 93.0) / 95.0)
        assert trade.net_pnl_pct > 0  # price fell further -> short profits

    def test_pnl_is_negative_when_price_rises_after_short_entry(self):
        closes = [100, 94, 96, 99, 102]
        opens = [100, 99, 95, 98, 101]
        df = _ohlc_df(closes, opens=opens)
        spec = StrategySpec(
            name="short_test", universe=["TEST"], start_date="2024-01-01", end_date="2024-01-05",
            entry=PctChangeCondition(field="close", lookback_days=1, operator="<", value=-0.05),
            exit=HoldingPeriodExit(days=1), direction="short",
            commission_pct=0.0, slippage_pct=0.0,
        )
        result = run_backtest(spec, {"TEST": df})
        trade = result.trades[0]
        assert trade.net_pnl_pct < 0  # price rose against the short -> loss

    def test_stop_loss_triggers_on_price_rise_not_fall(self):
        # adverse move for a short is a RISE — this must trip stop_loss,
        # not be mistaken for a favorable move.
        closes = [100, 100, 103, 90]
        df = _ohlc_df(closes)
        exit_rule = StopLossTakeProfitExit(stop_loss_pct=0.03, take_profit_pct=0.20, max_days=10)
        exit_i, exit_price, reason = _find_exit(exit_rule, df, entry_i=0, direction="short")
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(103.0)

    def test_take_profit_triggers_on_price_fall_not_rise(self):
        closes = [100, 100, 88, 120]
        df = _ohlc_df(closes)
        exit_rule = StopLossTakeProfitExit(stop_loss_pct=0.20, take_profit_pct=0.10, max_days=10)
        exit_i, exit_price, reason = _find_exit(exit_rule, df, entry_i=0, direction="short")
        assert reason == "take_profit"
        assert exit_price == pytest.approx(88.0)

    def test_long_stop_loss_logic_is_unaffected_by_the_direction_param(self):
        # regression: adding the direction param must not change existing
        # long behavior when direction defaults to "long"
        closes = [100, 100, 98, 95, 115]
        df = _ohlc_df(closes)
        exit_rule = StopLossTakeProfitExit(stop_loss_pct=0.03, take_profit_pct=0.08, max_days=10)
        exit_i, exit_price, reason = _find_exit(exit_rule, df, entry_i=0)
        assert reason == "stop_loss"
        assert exit_price == pytest.approx(95.0)

    def test_slippage_direction_is_reversed_for_short(self):
        closes = [100, 94, 93, 92, 91]
        opens = [100, 99, 95, 92, 91]
        df = _ohlc_df(closes, opens=opens)
        spec = StrategySpec(
            name="short_test", universe=["TEST"], start_date="2024-01-01", end_date="2024-01-05",
            entry=PctChangeCondition(field="close", lookback_days=1, operator="<", value=-0.05),
            exit=HoldingPeriodExit(days=1), direction="short",
            commission_pct=0.0, slippage_pct=0.02,
        )
        trade = run_backtest(spec, {"TEST": df}).trades[0]
        # short entry (a sell) is worsened DOWNWARD by slippage; exit (a buy
        # to cover) is worsened UPWARD — opposite of the long case.
        assert trade.entry_price == pytest.approx(95.0 * 0.98)
        assert trade.exit_price == pytest.approx(93.0 * 1.02)

    def test_direction_defaults_to_long(self):
        spec = StrategySpec(
            name="t", universe=["SBER"], start_date="2023-01-01", end_date="2023-06-01",
            entry=PctChangeCondition(field="close", lookback_days=1, operator=">", value=0.05),
            exit=HoldingPeriodExit(days=1),
        )
        assert spec.direction == "long"


class TestConditionExit:
    def test_exits_on_first_bar_matching_exit_condition(self):
        # exit when RSI(2) > 70 — construct a short rebound after entry
        closes = [100, 100, 101, 102, 108, 109]
        df = _ohlc_df(closes)
        exit_rule = ConditionExit(condition=RsiCondition(window=2, operator=">", value=70), max_days=5)
        exit_i, exit_price, reason = _find_exit(exit_rule, df, entry_i=0)
        assert reason == "condition"
        assert exit_i > 0

    def test_max_days_caps_condition_exit_when_never_satisfied(self):
        closes = [100, 99, 98, 97, 96, 95]  # steadily falling, RSI never exceeds 70
        df = _ohlc_df(closes)
        exit_rule = ConditionExit(condition=RsiCondition(window=2, operator=">", value=70), max_days=3)
        exit_i, exit_price, reason = _find_exit(exit_rule, df, entry_i=0)
        assert reason == "max_days"
        assert exit_i == 3


class TestSmaCrossCondition:
    def test_fires_only_on_the_crossing_bar_not_every_bar_above(self):
        # fast(2) stays flat at 10 while slow(4) is 10, then closes jump to 20 —
        # fast crosses above slow exactly once, then both stay "above" for a
        # while; the signal must be True only on the crossing day.
        closes = [10, 10, 10, 10, 20, 20, 20, 20, 20, 20]
        df = _ohlc_df(closes)
        cond = SmaCrossCondition(fast_window=2, slow_window=4, direction="above")
        signal = evaluate_condition(cond, df)
        assert signal.sum() == 1
        assert signal.iloc[4]
        assert not signal.iloc[5]
        assert not signal.iloc[6]

    def test_crossed_below_direction(self):
        # flat -> jump up (fast pulls above slow) -> flat (they converge) ->
        # sharp drop (fast dives below slow: the "crossed below" event)
        closes = [10, 10, 10, 10, 20, 20, 20, 20, 5, 5]
        df = _ohlc_df(closes)
        cond = SmaCrossCondition(fast_window=2, slow_window=4, direction="below")
        signal = evaluate_condition(cond, df)
        assert signal.sum() == 1
        assert signal.iloc[7]


class TestPriceVsSmaCondition:
    def test_price_above_sma(self):
        closes = [10, 10, 10, 10, 20]
        df = _ohlc_df(closes)
        cond = PriceVsSmaCondition(window=4, operator=">")
        signal = evaluate_condition(cond, df)
        assert not signal.iloc[3]  # 10 == sma(10) -> not strictly greater
        assert signal.iloc[4]  # 20 > sma(12.5)


class TestConsecutiveCandlesCondition:
    def test_signals_only_once_the_streak_reaches_count(self):
        opens = [100, 105, 100, 95, 98]
        closes = [105, 100, 95, 98, 90]
        # bar0 green, bar1 red, bar2 red (streak=2), bar3 green (breaks), bar4 red (streak=1)
        df = _ohlc_df(closes, opens=opens)
        cond = ConsecutiveCandlesCondition(count=2, candle_color="red")
        signal = evaluate_condition(cond, df)
        assert list(signal) == [False, False, True, False, False]

    def test_count_one_fires_on_every_matching_bar(self):
        opens = [100, 105, 100]
        closes = [105, 100, 95]  # green, red, red
        df = _ohlc_df(closes, opens=opens)
        cond = ConsecutiveCandlesCondition(count=1, candle_color="red")
        signal = evaluate_condition(cond, df)
        assert list(signal) == [False, True, True]

    def test_longer_streak_required_and_satisfied(self):
        opens = [100, 99, 98, 97, 96]
        closes = [99, 98, 97, 96, 95]  # 5 red bars in a row
        df = _ohlc_df(closes, opens=opens)
        cond = ConsecutiveCandlesCondition(count=3, candle_color="red")
        signal = evaluate_condition(cond, df)
        assert list(signal) == [False, False, True, True, True]

    def test_green_streak(self):
        opens = [100, 101, 102, 100]
        closes = [101, 102, 103, 99]  # green, green, green, red
        df = _ohlc_df(closes, opens=opens)
        cond = ConsecutiveCandlesCondition(count=3, candle_color="green")
        signal = evaluate_condition(cond, df)
        assert list(signal) == [False, False, True, False]

    def test_doji_bar_breaks_a_streak(self):
        opens = [100, 99, 98, 98]
        closes = [99, 98, 98, 97]  # red, red, doji (close==open), red
        df = _ohlc_df(closes, opens=opens)
        cond = ConsecutiveCandlesCondition(count=2, candle_color="red")
        signal = evaluate_condition(cond, df)
        # streak resets at the doji bar; only 1 red bar follows it, never reaches 2 again
        assert list(signal) == [False, True, False, False]


class TestMultiTickerPortfolio:
    def test_trades_sequenced_by_entry_date_and_capital_compounds(self):
        # Two independent tickers, one trade each, entered on different
        # dates. Documents the MVP simplification: trades are executed
        # sequentially in entry-date order against *current* equity,
        # regardless of whether their date ranges would really overlap.
        # opens deliberately differ from closes (as with real OHLC data) so
        # the entry executes at a distinct open price, not the signal bar's
        # own close.
        df_a = _ohlc_df([100, 106, 105, 108, 109], opens=[100, 101, 107, 108, 109])  # signal on bar 1 -> entry bar 2 @ open 107
        df_b = _ohlc_df([100, 100, 100, 106, 104], opens=[100, 100, 100, 100, 103])  # signal on bar 3 -> entry bar 4 @ open 103
        spec = StrategySpec(
            name="multi", universe=["A", "B"], start_date="2024-01-01", end_date="2024-01-05",
            entry=PctChangeCondition(field="close", lookback_days=1, operator=">", value=0.05),
            exit=HoldingPeriodExit(days=1), commission_pct=0.0, slippage_pct=0.0,
            initial_capital=1000.0,
        )
        result = run_backtest(spec, {"A": df_a, "B": df_b})
        assert len(result.trades) == 2
        assert result.trades[0].ticker == "A"  # earlier entry date
        assert result.trades[1].ticker == "B"

        # trade A: enter 107, exit 105 -> pnl = (105-107)/107
        expected_equity_after_a = 1000.0 * (1 + (105.0 - 107.0) / 107.0)
        assert result.equity_curve.iloc[1] == pytest.approx(expected_equity_after_a)

    def test_position_weight_scales_capital_at_risk(self):
        df = _ohlc_df([100, 106, 105, 108, 109], opens=[100, 101, 107, 108, 109])
        spec = StrategySpec(
            name="half", universe=["A"], start_date="2024-01-01", end_date="2024-01-05",
            entry=PctChangeCondition(field="close", lookback_days=1, operator=">", value=0.05),
            exit=HoldingPeriodExit(days=1), commission_pct=0.0, slippage_pct=0.0,
            initial_capital=1000.0, position_sizing=PositionSizing(weight=0.5),
        )
        result = run_backtest(spec, {"A": df})
        pnl_pct = (105.0 - 107.0) / 107.0
        expected_equity = 1000.0 + 1000.0 * 0.5 * pnl_pct
        assert result.equity_curve.iloc[-1] == pytest.approx(expected_equity)
