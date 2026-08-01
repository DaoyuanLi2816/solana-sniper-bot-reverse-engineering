import gzip
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from solana_sniper.paths import PROCESSED_DIR, RAW_DIR


def _instruction_features(row: dict) -> dict[str, object]:
    transaction = row.get("transaction") or {}
    message = transaction.get("message") or {}
    outer = message.get("instructions") or []
    meta = row.get("meta") or {}
    inner_groups = meta.get("innerInstructions") or []
    inner = [instruction for group in inner_groups for instruction in group.get("instructions", [])]
    parsed = [instruction.get("parsed", {}) for instruction in inner if instruction.get("parsed")]
    metadata = [
        item.get("info", {}) for item in parsed if item.get("type") == "initializeTokenMetadata"
    ]
    metadata_info = metadata[0] if metadata else {}
    account_keys = message.get("accountKeys") or []
    signatures = transaction.get("signatures") or []
    pre_balances = meta.get("preBalances") or []
    post_balances = meta.get("postBalances") or []
    signer_delta = None
    if pre_balances and post_balances:
        signer_delta = post_balances[0] - pre_balances[0]
    fee = meta.get("fee")
    priority_fee = None
    if fee is not None:
        priority_fee = max(0, fee - 5000 * max(1, len(signatures)))
    uri = str(metadata_info.get("uri") or "")
    name = str(metadata_info.get("name") or "")
    symbol = str(metadata_info.get("symbol") or "")
    return {
        "decision_time": row.get("blockTime"),
        "block_slot": row.get("slot"),
        "transaction_index": row.get("transactionIndex"),
        "tx_fee_lamports": fee,
        "priority_fee_lamports": priority_fee,
        "compute_units": meta.get("computeUnitsConsumed"),
        "cost_units": meta.get("costUnits"),
        "outer_instruction_count": len(outer),
        "inner_instruction_count": len(inner),
        "account_count": len(account_keys),
        "signature_count": len(signatures),
        "log_message_count": len(meta.get("logMessages") or []),
        "pre_token_balance_count": len(meta.get("preTokenBalances") or []),
        "post_token_balance_count": len(meta.get("postTokenBalances") or []),
        "signer_lamport_delta": signer_delta,
        "metadata_name_length": len(name),
        "metadata_symbol_length": len(symbol),
        "metadata_uri_length": len(uri),
        "metadata_uri_is_ipfs": int("ipfs" in uri.lower()),
        "metadata_present": int(bool(metadata_info)),
        "transaction_error": int(meta.get("err") is not None),
    }


def extract_jsonl_features(source: Path) -> pd.DataFrame:
    records = []
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            record = {"line_number": line_number, **_instruction_features(row)}
            records.append(record)
    return pd.DataFrame.from_records(records)


def build_positive_features() -> pd.DataFrame:
    raw = extract_jsonl_features(RAW_DIR / "core" / "bought_deploy_txs.jsonl.gz")
    index = pq.read_table(RAW_DIR / "core" / "bought_deploy_txs_index.parquet").to_pandas()
    merged = index.merge(raw, on="line_number", how="inner", validate="one_to_one")
    if len(merged) != len(index):
        raise ValueError(f"Feature extraction lost rows: {len(merged)} != {len(index)}")
    merged["label"] = 1
    merged["decision_time"] = pd.to_datetime(merged["decision_time"], unit="s", utc=True)
    merged["decision_hour_utc"] = merged["decision_time"].dt.hour
    merged["decision_weekday_utc"] = merged["decision_time"].dt.weekday
    return merged


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_positive_features()
    path = PROCESSED_DIR / "positive_deploy_features.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")
    print(
        json.dumps(
            {"path": str(path), "rows": len(frame), "columns": list(frame.columns)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
