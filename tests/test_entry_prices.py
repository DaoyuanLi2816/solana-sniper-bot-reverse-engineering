import duckdb
import pandas as pd
import pytest

from solana_sniper.entry_prices import selected_entry_sql, validate_entry_prices


def tiny_entry_frame() -> pd.DataFrame:
    connection = duckdb.connect()
    connection.execute(
        """
        CREATE TABLE universe(
            token_address VARCHAR,
            decision_time TIMESTAMPTZ,
            deploy_block_time BIGINT,
            deploy_block_slot BIGINT,
            deploy_tx_index BIGINT
        );
        INSERT INTO universe VALUES
            ('A', '2026-06-01T00:00:00Z', 1000, 100, 5),
            ('B', '2026-06-01T00:00:01Z', 1001, 200, 7);
        CREATE TABLE trades(
            token_address VARCHAR,
            slot_index_id VARCHAR,
            block_slot BIGINT,
            tx_index INTEGER,
            event_index INTEGER,
            block_time BIGINT,
            timestamp VARCHAR,
            tx_hash VARCHAR,
            side VARCHAR,
            program VARCHAR,
            price_usd DOUBLE,
            price_sol DOUBLE
        );
        INSERT INTO trades VALUES
            ('A', '0000000001000000060000', 100, 6, 0, 1000, 't0', 'h0', 'buy', 'pump', 2, 0.02),
            ('A', '0000000001010000010000', 101, 1, 0, 1001, 't1', 'h1', 'sell', 'pump', 3, 0.03),
            ('A', '0000000001030000020000', 103, 2, 0, 1003, 't3', 'h3', 'buy', 'pump', 4, 0.04);
        """
    )
    frame = connection.execute(selected_entry_sql("universe", "trades")).fetchdf()
    connection.close()
    return frame


def test_selects_first_trade_at_or_after_each_target_slot() -> None:
    frame = tiny_entry_frame()
    validate_entry_prices(frame, expected_tokens=2)
    token_a = frame[frame["token_address"] == "A"].sort_values("requested_delay_slots")
    assert token_a["entry_price_usd"].tolist() == [2.0, 3.0, 4.0]
    assert token_a["actual_delay_slots"].tolist() == [0, 1, 3]
    assert token_a["wait_slots_beyond_target"].tolist() == [0, 0, 1]
    assert not frame[frame["token_address"] == "B"]["covered"].any()


def test_validation_rejects_deployment_transaction_entry() -> None:
    frame = tiny_entry_frame()
    row = frame.index[(frame["token_address"] == "A") & (frame["requested_delay_slots"] == 0)][0]
    frame.loc[row, "entry_tx_index"] = frame.loc[row, "deploy_tx_index"]
    with pytest.raises(ValueError, match="deployment transaction"):
        validate_entry_prices(frame, expected_tokens=2)
