from pathlib import Path

import pandas as pd

from solana_sniper.stream_negative_sample import join_index_and_features


def test_join_index_uses_canonical_utc_time_parts(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.parquet"
    index_path = tmp_path / "index.parquet"
    output_path = tmp_path / "output.parquet"
    pd.DataFrame(
        {
            "line_number": [25, 50],
            "decision_time": [1_767_225_659, 1_770_000_000],
            "block_slot": [10, 20],
            "transaction_index": [1, 2],
            "tx_fee_lamports": [5_000, 5_000],
            "priority_fee_lamports": [0, 1_000],
            "compute_units": [100_000, 200_000],
            "cost_units": [100_000, 200_000],
            "outer_instruction_count": [1, 1],
            "inner_instruction_count": [1, 1],
            "account_count": [10, 10],
            "signature_count": [1, 1],
            "log_message_count": [2, 2],
            "pre_token_balance_count": [0, 0],
            "post_token_balance_count": [1, 1],
            "signer_lamport_delta": [-10_000, -20_000],
            "metadata_name_length": [4, 5],
            "metadata_symbol_length": [3, 4],
            "metadata_uri_length": [40, 41],
            "metadata_uri_is_ipfs": [1, 0],
            "metadata_present": [1, 1],
            "transaction_error": [0, 0],
        }
    ).to_parquet(raw_path, index=False)
    pd.DataFrame(
        {
            "line_number": [25, 50],
            "tx_hash": ["tx1", "tx2"],
            "blockTime": [1_767_225_659, 1_770_000_000],
            "blockSlot": [10, 20],
            "token_address": ["token1", "token2"],
            "tx_signer": ["signer1", "signer2"],
            "creator_address": ["creator1", "creator2"],
        }
    ).to_parquet(index_path, index=False)

    result = join_index_and_features(
        25,
        raw_features=raw_path,
        index=index_path,
        output=output_path,
    )
    frame = pd.read_parquet(result)
    canonical = pd.to_datetime(frame["decision_time"], utc=True)
    assert frame["decision_hour_utc"].tolist() == canonical.dt.hour.tolist()
    assert frame["decision_weekday_utc"].tolist() == canonical.dt.weekday.tolist()
    assert frame["token_address"].tolist() == ["token1", "token2"]
