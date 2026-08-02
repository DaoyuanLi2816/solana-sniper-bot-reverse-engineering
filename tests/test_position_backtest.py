import pandas as pd

from solana_sniper.position_backtest import (
    build_position_entries,
    training_position_parameters,
    validate_position_entries,
)


def test_training_position_parameters_respect_cutoff_and_same_slot() -> None:
    latency = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-02-01T00:00:00Z"]
            ),
            "latency_slots": [0, 1, 0],
            "same_slot_position_delta": [12, 999, 500],
        }
    )
    result = training_position_parameters(latency, pd.Timestamp("2026-01-31T00:00:00Z"))

    assert result["rows"] == 1
    assert result["median"] == 12
    assert result["maximum"] == 12


def test_position_entry_requires_same_slot_and_position_floor(tmp_path) -> None:
    candidates = pd.DataFrame(
        {
            "token_address": ["a", "b"],
            "decision_time": pd.to_datetime(["2026-01-01T00:00:00Z"] * 2),
            "label": [0, 1],
            "deploy_block_time": [100, 100],
            "deploy_block_slot": [10, 10],
            "deploy_tx_index": [5, 5],
            "probability": [0.9, 0.9],
            "population_weight": [25.0, 1.0],
        }
    )
    trades = pd.DataFrame(
        {
            "token_address": ["a", "a", "a", "b"],
            "slot_index_id": [1, 2, 3, 4],
            "block_slot": [10, 10, 11, 11],
            "tx_index": [6, 17, 0, 20],
            "event_index": [0, 0, 0, 0],
            "block_time": [100, 100, 101, 101],
            "timestamp": [100, 100, 101, 101],
            "tx_hash": ["a1", "a2", "a3", "b1"],
            "side": ["buy"] * 4,
            "program": ["pump"] * 4,
            "price_usd": [1.0, 2.0, 3.0, 4.0],
            "price_sol": [0.01, 0.02, 0.03, 0.04],
        }
    )
    trades_path = tmp_path / "trades.parquet"
    trades.to_parquet(trades_path, index=False)

    result = build_position_entries(
        candidates, trades_path, position_lag=12, outcome_cutoff_epoch=200
    )
    validate_position_entries(result, 12)

    a = result.loc[result["token_address"] == "a"].iloc[0]
    b = result.loc[result["token_address"] == "b"].iloc[0]
    assert bool(a["covered"])
    assert a["entry_tx_index"] == 17
    assert a["actual_position_delta"] == 12
    assert not bool(b["covered"])
