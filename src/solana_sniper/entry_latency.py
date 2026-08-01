import gzip
import json

import pandas as pd
import pyarrow.parquet as pq

from solana_sniper.audit_wallet import _finite_summary
from solana_sniper.paths import PROCESSED_DIR, RAW_DIR, REPORT_DIR


def _wallet_transaction_positions() -> pd.DataFrame:
    records = []
    source = RAW_DIR / "wallet" / "5brv79e_activity_txs.jsonl.gz"
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            records.append(
                {
                    "line_number": line_number,
                    "buy_transaction_index": row.get("transactionIndex"),
                }
            )
    positions = pd.DataFrame.from_records(records)
    index = pq.read_table(RAW_DIR / "wallet" / "5brv79e_activity_txs_index.parquet").to_pandas()
    return index.merge(positions, on="line_number", how="inner", validate="one_to_one")


def build_latency_table() -> pd.DataFrame:
    activity = pq.read_table(RAW_DIR / "wallet" / "5brv79e_activity.parquet").to_pandas()
    buys = activity[activity["event_type"] == "buy"].sort_values("timestamp", kind="stable")
    first_buys = buys.drop_duplicates("token_address", keep="first")[
        ["token_address", "tx_hash", "timestamp", "cost_usd"]
    ].rename(
        columns={
            "tx_hash": "buy_tx_hash",
            "timestamp": "wallet_buy_timestamp",
            "cost_usd": "entry_cost_usd",
        }
    )
    wallet_positions = _wallet_transaction_positions().rename(
        columns={"tx_hash": "buy_tx_hash", "blockTime": "buy_block_time", "blockSlot": "buy_slot"}
    )
    first_buys = first_buys.merge(
        wallet_positions[["buy_tx_hash", "buy_block_time", "buy_slot", "buy_transaction_index"]],
        on="buy_tx_hash",
        how="left",
        validate="one_to_one",
    )
    deployments = pd.read_parquet(PROCESSED_DIR / "positive_deploy_features.parquet")
    deployments = deployments[
        ["token_address", "tx_hash", "decision_time", "block_slot", "transaction_index"]
    ].rename(
        columns={
            "tx_hash": "deploy_tx_hash",
            "block_slot": "deploy_slot",
            "transaction_index": "deploy_transaction_index",
        }
    )
    result = deployments.merge(first_buys, on="token_address", how="left", validate="one_to_one")
    result["entry_cost_usd"] = pd.to_numeric(result["entry_cost_usd"], errors="coerce")
    deploy_epoch_seconds = pd.to_datetime(result["decision_time"], utc=True).map(
        lambda value: int(value.timestamp())
    )
    result["latency_seconds"] = result["buy_block_time"] - deploy_epoch_seconds
    result["latency_slots"] = result["buy_slot"] - result["deploy_slot"]
    same_slot = result["latency_slots"] == 0
    result["same_slot_position_delta"] = pd.NA
    result.loc[same_slot, "same_slot_position_delta"] = (
        result.loc[same_slot, "buy_transaction_index"]
        - result.loc[same_slot, "deploy_transaction_index"]
    )
    return result


def main() -> None:
    table = build_latency_table()
    output = PROCESSED_DIR / "entry_latency.parquet"
    table.to_parquet(output, index=False)
    valid = table[table["latency_slots"].notna()]
    summary = {
        "deployment_rows": int(len(table)),
        "matched_first_buys": int(len(valid)),
        "zero_slot_share": float((valid["latency_slots"] == 0).mean()),
        "latency_slots": _finite_summary(valid["latency_slots"]),
        "latency_seconds": _finite_summary(valid["latency_seconds"]),
        "same_slot_position_delta": _finite_summary(valid["same_slot_position_delta"]),
        "entry_cost_usd": _finite_summary(valid["entry_cost_usd"]),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORT_DIR / "entry_latency.json"
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
