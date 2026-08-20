import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from strategy_spec import (  # noqa: E402
    ConditionExit,
    ConsecutiveCandlesCondition,
    HoldingPeriodExit,
    PctChangeCondition,
    RsiCondition,
    StopLossTakeProfitExit,
    StrategySpec,
)


def _base_kwargs(**overrides):
    kwargs = dict(
        name="test",
        universe=["SBER"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        entry=PctChangeCondition(operator=">", value=0.05),
        exit=HoldingPeriodExit(days=1),
    )
    kwargs.update(overrides)
    return kwargs


class TestStrategySpecValidation:
    def test_valid_spec_constructs(self):
        spec = StrategySpec(**_base_kwargs())
        assert spec.universe == ["SBER"]
        assert spec.commission_pct == pytest.approx(0.0005)  # default

    def test_end_date_before_start_date_rejected(self):
        with pytest.raises(ValidationError):
            StrategySpec(**_base_kwargs(start_date="2023-12-31", end_date="2023-01-01"))

    def test_end_date_equal_start_date_rejected(self):
        with pytest.raises(ValidationError):
            StrategySpec(**_base_kwargs(start_date="2023-01-01", end_date="2023-01-01"))

    def test_empty_universe_rejected(self):
        with pytest.raises(ValidationError):
            StrategySpec(**_base_kwargs(universe=[]))

    def test_negative_commission_rejected(self):
        with pytest.raises(ValidationError):
            StrategySpec(**_base_kwargs(commission_pct=-0.01))

    def test_zero_initial_capital_rejected(self):
        with pytest.raises(ValidationError):
            StrategySpec(**_base_kwargs(initial_capital=0))

    def test_position_weight_over_one_rejected(self):
        from strategy_spec import PositionSizing

        with pytest.raises(ValidationError):
            PositionSizing(weight=1.5)

    def test_defaults_are_sane(self):
        spec = StrategySpec(**_base_kwargs())
        assert spec.slippage_pct == 0.0
        assert spec.initial_capital == pytest.approx(1_000_000.0)
        assert spec.position_sizing.type == "equal_weight"
        assert spec.position_sizing.weight == pytest.approx(1.0)
        assert spec.direction == "long"

    def test_short_direction_accepted(self):
        spec = StrategySpec(**_base_kwargs(direction="short"))
        assert spec.direction == "short"

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValidationError):
            StrategySpec(**_base_kwargs(direction="sideways"))


class TestConditionValidation:
    def test_invalid_operator_rejected(self):
        with pytest.raises(ValidationError):
            PctChangeCondition(operator="==", value=0.05)  # type: ignore[arg-type]

    def test_rsi_value_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            RsiCondition(operator=">", value=150)

    def test_rsi_defaults(self):
        cond = RsiCondition(operator="<", value=30)
        assert cond.window == 14

    def test_consecutive_candles_valid(self):
        cond = ConsecutiveCandlesCondition(count=3, candle_color="red")
        assert cond.count == 3
        assert cond.candle_color == "red"

    def test_consecutive_candles_count_must_be_positive(self):
        with pytest.raises(ValidationError):
            ConsecutiveCandlesCondition(count=0, candle_color="green")

    def test_consecutive_candles_invalid_color_rejected(self):
        with pytest.raises(ValidationError):
            ConsecutiveCandlesCondition(count=1, candle_color="blue")  # type: ignore[arg-type]


class TestExitValidation:
    def test_stop_loss_take_profit_needs_at_least_one_bound(self):
        with pytest.raises(ValidationError):
            StopLossTakeProfitExit()

    def test_stop_loss_take_profit_accepts_single_bound(self):
        exit_rule = StopLossTakeProfitExit(stop_loss_pct=0.05)
        assert exit_rule.take_profit_pct is None

    def test_holding_period_requires_positive_days(self):
        with pytest.raises(ValidationError):
            HoldingPeriodExit(days=0)

    def test_condition_exit_wraps_a_condition(self):
        exit_rule = ConditionExit(condition=RsiCondition(operator=">", value=70), max_days=10)
        assert exit_rule.condition.type == "rsi"


class TestSerialization:
    def test_round_trip_through_json(self):
        spec = StrategySpec(**_base_kwargs())
        raw = spec.model_dump_json()
        rebuilt = StrategySpec.model_validate_json(raw)
        assert rebuilt == spec

    def test_entry_type_discriminates_correctly_from_json(self):
        raw = """{
            "name": "t", "universe": ["SBER"],
            "start_date": "2023-01-01", "end_date": "2023-06-01",
            "entry": {"type": "rsi", "window": 14, "operator": "<", "value": 30},
            "exit": {"type": "holding_period", "days": 5}
        }"""
        spec = StrategySpec.model_validate_json(raw)
        assert spec.entry.type == "rsi"
        assert spec.entry.window == 14
