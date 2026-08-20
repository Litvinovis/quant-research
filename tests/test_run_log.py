import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from run_log import log_run  # noqa: E402


def _spec():
    return {
        "name": "t",
        "universe": ["SBER"],
        "start_date": "2023-01-01",
        "end_date": "2023-06-01",
        "entry": {"type": "pct_change", "field": "close", "lookback_days": 1, "operator": ">", "value": 0.05},
        "exit": {"type": "holding_period", "days": 1},
        "commission_pct": 0.0005,
        "position_sizing": {"type": "equal_weight", "weight": 1.0},
    }


def _metrics():
    return {
        "num_trades": 3,
        "total_return_pct": 1.5,
        "total_return_pct_gross": 2.3,
        "cagr_pct": 4.2,
        "cagr_pct_gross": 6.1,
        "max_drawdown_pct": -3.1,
        "sharpe": 0.8,
        "win_rate_pct": 66.6,
        "avg_trade_pnl_pct": 0.5,
    }


def _mock_conn_returning_run_id(run_id=42):
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (run_id,)
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    return mock_conn, mock_cursor


class TestLogRun:
    def test_inserts_backtest_runs_row(self):
        mock_conn, mock_cursor = _mock_conn_returning_run_id()
        with patch("run_log.psycopg.connect", return_value=mock_conn) as mock_connect:
            log_run(_spec(), _metrics())

        mock_connect.assert_called_once()
        assert mock_cursor.execute.call_count == 2  # backtest_runs insert + strategies upsert
        first_call_sql = mock_cursor.execute.call_args_list[0][0][0]
        first_call_params = mock_cursor.execute.call_args_list[0][0][1]
        assert "INSERT INTO backtest_runs" in first_call_sql
        assert first_call_params[0] == "t"  # spec_name
        assert first_call_params[5] == 3  # num_trades
        assert first_call_params[7] == 2.3  # total_return_pct_gross
        assert first_call_params[9] == 6.1  # cagr_pct_gross

    def test_upserts_strategies_catalog(self):
        mock_conn, mock_cursor = _mock_conn_returning_run_id(run_id=99)
        with patch("run_log.psycopg.connect", return_value=mock_conn):
            log_run(_spec(), _metrics())

        second_call_sql = mock_cursor.execute.call_args_list[1][0][0]
        second_call_params = mock_cursor.execute.call_args_list[1][0][1]
        assert "INSERT INTO strategies" in second_call_sql
        assert "ON CONFLICT (name) DO UPDATE" in second_call_sql
        assert second_call_params[0] == "t"  # name
        assert second_call_params[4] == "long"  # direction, default from spec.get()
        assert second_call_params[6] == 1.0  # position_weight
        assert second_call_params[7] == 99  # last_backtest_run_id from RETURNING id
        assert second_call_params[10] == 2.3  # total_return_pct_gross
        assert second_call_params[12] == 6.1  # cagr_pct_gross

    def test_short_direction_is_persisted(self):
        mock_conn, mock_cursor = _mock_conn_returning_run_id()
        spec = _spec()
        spec["direction"] = "short"
        with patch("run_log.psycopg.connect", return_value=mock_conn):
            log_run(spec, _metrics())

        second_call_params = mock_cursor.execute.call_args_list[1][0][1]
        assert second_call_params[4] == "short"

    def test_db_error_is_swallowed_not_raised(self):
        with patch("run_log.psycopg.connect", side_effect=RuntimeError("db unreachable")):
            log_run(_spec(), _metrics())  # must not raise
