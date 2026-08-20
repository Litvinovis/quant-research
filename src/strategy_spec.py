"""
Strategy specification schema — the DSL the AI layer fills in from natural
language. The engine (engine.py) only ever interprets this fixed vocabulary;
it never executes freeform code, so a given spec always backtests to the
same numeric result.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Operator = Literal[">", ">=", "<", "<="]


class PctChangeCondition(BaseModel):
    type: Literal["pct_change"] = "pct_change"
    field: Literal["close", "open", "high", "low"] = "close"
    lookback_days: int = Field(ge=1, default=1)
    operator: Operator
    value: float  # fraction, e.g. 0.05 = 5%


class SmaCrossCondition(BaseModel):
    type: Literal["sma_cross"] = "sma_cross"
    fast_window: int = Field(ge=1)
    slow_window: int = Field(ge=1)
    direction: Literal["above", "below"]  # fast crosses above/below slow


class RsiCondition(BaseModel):
    type: Literal["rsi"] = "rsi"
    window: int = Field(ge=2, default=14)
    operator: Operator
    value: float = Field(ge=0, le=100)


class PriceVsSmaCondition(BaseModel):
    type: Literal["price_vs_sma"] = "price_vs_sma"
    window: int = Field(ge=1)
    operator: Operator


class ConsecutiveCandlesCondition(BaseModel):
    """True on bar T if the last `count` bars (T-count+1 .. T) are ALL the
    same color — bar-timeframe-agnostic, so this is what "N red 5-minute
    candles in a row" or "N red daily candles in a row" both compile to."""
    type: Literal["consecutive_candles"] = "consecutive_candles"
    count: int = Field(ge=1)
    candle_color: Literal["green", "red"]  # green: close > open, red: close < open


Condition = (
    PctChangeCondition | SmaCrossCondition | RsiCondition | PriceVsSmaCondition | ConsecutiveCandlesCondition
)


class HoldingPeriodExit(BaseModel):
    type: Literal["holding_period"] = "holding_period"
    days: int = Field(ge=1)


class StopLossTakeProfitExit(BaseModel):
    type: Literal["stop_loss_take_profit"] = "stop_loss_take_profit"
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_days: Optional[int] = None

    @model_validator(mode="after")
    def _at_least_one_bound(self):
        if self.stop_loss_pct is None and self.take_profit_pct is None and self.max_days is None:
            raise ValueError("stop_loss_take_profit exit needs at least one of stop_loss_pct/take_profit_pct/max_days")
        return self


class ConditionExit(BaseModel):
    type: Literal["condition"] = "condition"
    condition: Condition
    max_days: Optional[int] = None  # safety cap, avoid unbounded holds


ExitRule = HoldingPeriodExit | StopLossTakeProfitExit | ConditionExit


class PositionSizing(BaseModel):
    type: Literal["equal_weight"] = "equal_weight"
    weight: float = Field(gt=0, le=1, default=1.0)  # fraction of capital per open position


class StrategySpec(BaseModel):
    name: str
    universe: list[str] = Field(min_length=1)
    start_date: date
    end_date: date
    entry: Condition
    exit: ExitRule
    # "short" flips P&L sign and stop_loss_take_profit's trigger direction
    # (see engine.py). Does NOT model borrow fees or short availability —
    # a real short has both, and this backtest doesn't charge for either.
    direction: Literal["long", "short"] = "long"
    position_sizing: PositionSizing = PositionSizing()
    commission_pct: float = Field(ge=0, default=0.0005)  # 0.05% per side, typical MOEX broker
    slippage_pct: float = Field(ge=0, default=0.0)
    initial_capital: float = Field(gt=0, default=1_000_000.0)

    @model_validator(mode="after")
    def _dates_ordered(self):
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self
