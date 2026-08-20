"""
Deterministic backtest engine. Interprets a StrategySpec against historical
OHLCV data and produces trades + an equity curve. No LLM involvement here —
same spec + same data always produces the same numbers.

Execution model (documented so results are auditable):
- A signal computed using data through bar T is acted on at bar T+1's open
  (no lookahead — you can't trade on a bar using its own close as if it
  were known beforehand).
- `holding_period` exit: position closed at the CLOSE of bar
  (entry_bar + days - 1). E.g. days=1 means: enter next bar's open, exit
  that same bar's close.
- `stop_loss_take_profit` / `condition` exits: evaluated bar-by-bar from the
  entry bar onward; whichever condition is hit first (in bar order) closes
  the trade. `max_days` is a safety cap so no exit rule can hold forever.
- One open position per ticker at a time (no pyramiding); overlapping
  entries on the same ticker are skipped.
- Portfolio-level, trades are sequenced in chronological order of entry
  across the whole universe, each risking `weight` fraction of *current*
  equity at the time it opens (compounding, no simultaneous overlapping
  trades across tickers in this MVP — see README for the rationale).
- `spec.direction`: "long" (default) or "short". Short flips the P&L sign
  and stop_loss_take_profit's trigger direction (a price rise is now the
  adverse move) and the slippage direction on both legs. Does not model
  borrow fees or short availability — see strategy_spec.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from indicators import pct_change, rsi, sma
from strategy_spec import (
    Condition,
    ConditionExit,
    ConsecutiveCandlesCondition,
    ExitRule,
    HoldingPeriodExit,
    PctChangeCondition,
    PriceVsSmaCondition,
    RsiCondition,
    SmaCrossCondition,
    StopLossTakeProfitExit,
    StrategySpec,
)


@dataclass
class Trade:
    ticker: str
    entry_date: object
    entry_price: float
    exit_date: object
    exit_price: float
    exit_reason: str
    gross_pnl_pct: float  # price move net of slippage, before commission
    net_pnl_pct: float  # gross_pnl_pct minus commission — this is what you actually keep


@dataclass
class BacktestResult:
    spec_name: str
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)  # compounds net_pnl_pct
    gross_equity_curve: pd.Series = field(default_factory=pd.Series)  # compounds gross_pnl_pct — "what if commission were 0"


def evaluate_condition(cond: Condition, df: pd.DataFrame) -> pd.Series:
    """Boolean series aligned to df.index. True on bar T means the
    condition, using only data through bar T, is satisfied."""
    if isinstance(cond, PctChangeCondition):
        values = pct_change(df[cond.field], cond.lookback_days)
        return _apply_operator(values, cond.operator, cond.value)

    if isinstance(cond, SmaCrossCondition):
        fast = sma(df["close"], cond.fast_window)
        slow = sma(df["close"], cond.slow_window)
        above = fast > slow
        # shift() introduces a leading NaN, which upcasts the bool Series to
        # object dtype; without astype(bool), `~` on that object dtype does
        # bitwise-not on Python int/bool objects instead of logical negation.
        prev_above = above.shift(1).fillna(False).astype(bool)
        crossed_above = above & ~prev_above
        crossed_below = (~above) & prev_above
        return crossed_above if cond.direction == "above" else crossed_below

    if isinstance(cond, RsiCondition):
        values = rsi(df["close"], cond.window)
        return _apply_operator(values, cond.operator, cond.value)

    if isinstance(cond, PriceVsSmaCondition):
        values = df["close"] - sma(df["close"], cond.window)
        return _apply_operator(values, cond.operator, 0.0)

    if isinstance(cond, ConsecutiveCandlesCondition):
        # doji bars (close == open) count as neither color, breaking a streak either way
        target = df["close"] > df["open"] if cond.candle_color == "green" else df["close"] < df["open"]
        # rolling sum of `count` consecutive True bars == count -> all True in the window
        streak = target.astype(int).rolling(window=cond.count).sum()
        return (streak == cond.count).fillna(False)

    raise ValueError(f"Unknown condition type: {cond}")


def _apply_operator(values: pd.Series, operator: str, threshold: float) -> pd.Series:
    ops = {
        ">": values > threshold,
        ">=": values >= threshold,
        "<": values < threshold,
        "<=": values <= threshold,
    }
    return ops[operator].fillna(False)


def _find_exit(exit_rule: ExitRule, df: pd.DataFrame, entry_i: int, direction: str = "long") -> tuple[int, float, str]:
    n = len(df)
    entry_price = df["open"].iloc[entry_i]

    if isinstance(exit_rule, HoldingPeriodExit):
        exit_i = min(entry_i + exit_rule.days - 1, n - 1)
        return exit_i, df["close"].iloc[exit_i], "holding_period"

    if isinstance(exit_rule, StopLossTakeProfitExit):
        max_i = min(entry_i + (exit_rule.max_days or n), n - 1)
        for i in range(entry_i, max_i + 1):
            change = (df["close"].iloc[i] - entry_price) / entry_price
            # long: adverse move is a price drop, favorable is a rise.
            # short: adverse move is a price rise, favorable is a drop.
            if direction == "long":
                stop_hit = exit_rule.stop_loss_pct is not None and change <= -exit_rule.stop_loss_pct
                profit_hit = exit_rule.take_profit_pct is not None and change >= exit_rule.take_profit_pct
            else:
                stop_hit = exit_rule.stop_loss_pct is not None and change >= exit_rule.stop_loss_pct
                profit_hit = exit_rule.take_profit_pct is not None and change <= -exit_rule.take_profit_pct
            if stop_hit:
                return i, df["close"].iloc[i], "stop_loss"
            if profit_hit:
                return i, df["close"].iloc[i], "take_profit"
        return max_i, df["close"].iloc[max_i], "max_days"

    if isinstance(exit_rule, ConditionExit):
        signal = evaluate_condition(exit_rule.condition, df)
        max_i = min(entry_i + (exit_rule.max_days or n), n - 1)
        for i in range(entry_i + 1, max_i + 1):
            if signal.iloc[i]:
                return i, df["close"].iloc[i], "condition"
        return max_i, df["close"].iloc[max_i], "max_days"

    raise ValueError(f"Unknown exit rule type: {exit_rule}")


def _collect_trades(ticker: str, df: pd.DataFrame, spec: StrategySpec) -> list[Trade]:
    signal = evaluate_condition(spec.entry, df)
    n = len(df)
    trades = []
    i = 0
    while i < n - 1:
        if signal.iloc[i]:
            entry_i = i + 1
            if entry_i >= n:
                break
            # Long opens by buying (slippage pushes the fill up) and closes
            # by selling (slippage pushes it down). Short opens by selling
            # (slippage pushes it down) and closes by buying to cover
            # (slippage pushes it up) — the opposite of long on both legs.
            if spec.direction == "long":
                entry_price = df["open"].iloc[entry_i] * (1 + spec.slippage_pct)
            else:
                entry_price = df["open"].iloc[entry_i] * (1 - spec.slippage_pct)
            exit_i, raw_exit_price, reason = _find_exit(spec.exit, df, entry_i, spec.direction)
            if spec.direction == "long":
                exit_price = raw_exit_price * (1 - spec.slippage_pct)
            else:
                exit_price = raw_exit_price * (1 + spec.slippage_pct)

            if spec.direction == "long":
                gross_pct = (exit_price - entry_price) / entry_price
            else:
                gross_pct = (entry_price - exit_price) / entry_price
            net_pct = gross_pct - 2 * spec.commission_pct  # entry + exit commission

            trades.append(
                Trade(
                    ticker=ticker,
                    entry_date=df.index[entry_i],
                    entry_price=entry_price,
                    exit_date=df.index[exit_i],
                    exit_price=exit_price,
                    exit_reason=reason,
                    gross_pnl_pct=gross_pct,
                    net_pnl_pct=net_pct,
                )
            )
            i = exit_i + 1
        else:
            i += 1
    return trades


def _filter_by_date_range(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """df.index is `date` objects for daily bars (data.py::fetch_candles) or
    a tz-aware DatetimeIndex for intraday bars (market_data-style fetchers).
    Comparing a datetime64 DatetimeIndex to a plain `date` scalar raises
    TypeError in pandas; comparing an object-dtype date Index to a
    Timestamp scalar raises too — each representation needs its own bound
    type, so branch on it rather than picking one and breaking the other."""
    if pd.api.types.is_datetime64_any_dtype(df.index):
        start_bound = pd.Timestamp(start)
        end_bound = pd.Timestamp(end) + pd.Timedelta(days=1)
        if df.index.tz is not None:
            start_bound = start_bound.tz_localize(df.index.tz)
            end_bound = end_bound.tz_localize(df.index.tz)
        return df[(df.index >= start_bound) & (df.index < end_bound)]
    return df[(df.index >= start) & (df.index <= end)]


def run_backtest(spec: StrategySpec, data: dict[str, pd.DataFrame]) -> BacktestResult:
    all_trades: list[Trade] = []
    for ticker in spec.universe:
        df = _filter_by_date_range(data[ticker], spec.start_date, spec.end_date)
        if df.empty:
            continue
        all_trades.extend(_collect_trades(ticker, df, spec))

    all_trades.sort(key=lambda t: t.entry_date)

    equity_curve = _build_equity_curve(spec, all_trades, lambda t: t.net_pnl_pct)
    gross_equity_curve = _build_equity_curve(spec, all_trades, lambda t: t.gross_pnl_pct)
    return BacktestResult(
        spec_name=spec.name, trades=all_trades,
        equity_curve=equity_curve, gross_equity_curve=gross_equity_curve,
    )


def _to_utc_timestamp(d) -> pd.Timestamp:
    """spec.start_date is always a naive `date`; trade.exit_date is a naive
    `date` for daily bars but a tz-aware Timestamp for intraday bars (see
    _filter_by_date_range). Mixing naive and tz-aware values in one
    pd.to_datetime() call raises — normalize everything to UTC first."""
    ts = pd.Timestamp(d)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _build_equity_curve(spec: StrategySpec, trades: list[Trade], pnl_of) -> pd.Series:
    """Independent compounding pass — gross and net equity are tracked
    separately (not derived from each other) because the stake at each
    step is a fraction of *that* curve's own current equity."""
    equity = spec.initial_capital
    dates = [_to_utc_timestamp(spec.start_date)]
    values = [equity]
    for trade in trades:
        stake = equity * spec.position_sizing.weight
        equity += stake * pnl_of(trade)
        dates.append(_to_utc_timestamp(trade.exit_date))
        values.append(equity)
    return pd.Series(values, index=pd.DatetimeIndex(dates)).groupby(level=0).last()
