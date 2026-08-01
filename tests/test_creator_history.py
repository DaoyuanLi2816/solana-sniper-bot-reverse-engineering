from pathlib import Path

import pandas as pd

from solana_sniper.creator_history import build_creator_history_dataset


def _write_index(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_creator_history_excludes_same_slot_and_future(tmp_path: Path) -> None:
    base = {
        "tx_hash": "",
        "line_number": 0,
        "blockTime": 0,
        "blockSlot": 0,
        "token_address": "",
        "tx_signer": "creator-a",
        "creator_address": None,
    }
    negative_rows = []
    for line, slot, timestamp, token in [
        (1, 10, 100, "token-1"),
        (2, 20, 200, "token-2"),
        (3, 30, 200, "token-3"),
        (3, 30, 200, "token-4"),
    ]:
        negative_rows.append(
            {
                **base,
                "tx_hash": f"tx-{token}",
                "line_number": line,
                "blockTime": timestamp,
                "blockSlot": slot,
                "token_address": token,
            }
        )
    positive_path = tmp_path / "positive.parquet"
    negative_path = tmp_path / "negative.parquet"
    _write_index(
        positive_path,
        [
            {
                **base,
                "tx_hash": "tx-unrelated",
                "line_number": 1,
                "blockTime": 150,
                "blockSlot": 15,
                "token_address": "token-unrelated",
                "tx_signer": "creator-b",
            }
        ],
    )
    _write_index(negative_path, negative_rows)
    classification = pd.DataFrame(
        {
            "tx_hash": [row["tx_hash"] for row in negative_rows],
            "token_address": [row["token_address"] for row in negative_rows],
            "tx_signer": ["creator-a"] * 4,
            "blockSlot": [10, 20, 30, 30],
            "decision_time": pd.to_datetime([100, 200, 200, 200], unit="s", utc=True),
            "label": [0, 0, 0, 0],
        }
    )
    classification_path = tmp_path / "classification.parquet"
    output_path = tmp_path / "output.parquet"
    classification.to_parquet(classification_path, index=False)
    result = build_creator_history_dataset(
        classification_path, positive_path, negative_path, output_path
    )
    output = pd.read_parquet(output_path).sort_values("token_address")
    assert result["strict_time_violations"] == 0
    assert output["creator_prior_deploy_count"].tolist() == [0, 1, 2, 2]
    assert output["creator_seconds_since_previous_deploy"].tolist()[1:] == [100.0, 0.0, 0.0]
