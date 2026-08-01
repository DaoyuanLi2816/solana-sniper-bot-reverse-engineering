import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from solana_sniper.audit_trades_metadata import EXPECTED_ROWS, validate_metadata
from solana_sniper.manifest import sha256_file
from solana_sniper.paths import PROJECT_ROOT, RAW_DIR, REPORT_DIR

SOURCE_PATH = RAW_DIR / "june" / "pumpfun_trades.parquet"
OUTPUT_PATH = REPORT_DIR / "trades_source_audit.json"


def fetch_dict(connection: duckdb.DuckDBPyConnection, query: str) -> dict[str, object]:
    cursor = connection.execute(query)
    row = cursor.fetchone()
    if row is None:
        raise ValueError("Audit query unexpectedly returned no rows")
    return dict(zip((column[0] for column in cursor.description), row, strict=True))


def audit_source(source: Path = SOURCE_PATH) -> dict[str, object]:
    if not source.exists():
        raise FileNotFoundError(source)
    parquet = pq.ParquetFile(source)
    validate_metadata(parquet.schema_arrow, parquet.metadata.num_rows, source.stat().st_size)

    temporary = PROJECT_ROOT / "data" / "processed" / "duckdb_trades_audit_tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    parquet_sql_path = source.as_posix().replace("'", "''")
    relation = f"read_parquet('{parquet_sql_path}')"
    null_terms = [f'count(*) - count("{name}") AS "{name}"' for name in parquet.schema_arrow.names]

    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '12GB'")
    connection.execute("SET preserve_insertion_order = false")
    temp_sql_path = temporary.as_posix().replace("'", "''")
    connection.execute(f"SET temp_directory = '{temp_sql_path}'")
    summary = fetch_dict(
        connection,
        f"""
        SELECT
            count(*) AS rows,
            count(DISTINCT token_address) AS unique_tokens,
            min(block_time) AS min_block_time,
            max(block_time) AS max_block_time,
            min(block_slot) AS min_block_slot,
            max(block_slot) AS max_block_slot,
            count(*) FILTER (
                WHERE block_slot < deploy_block_slot
                   OR (block_slot = deploy_block_slot AND tx_index < deploy_tx_index)
            ) AS pre_deploy_position_rows,
            count(*) FILTER (
                WHERE block_slot = deploy_block_slot AND tx_index = deploy_tx_index
            ) AS same_deploy_transaction_rows,
            count(*) FILTER (
                WHERE block_slot = deploy_block_slot AND tx_index > deploy_tx_index
            ) AS executable_same_slot_rows,
            count(DISTINCT token_address) FILTER (
                WHERE block_slot = deploy_block_slot AND tx_index > deploy_tx_index
            ) AS executable_same_slot_tokens,
            count(*) FILTER (WHERE block_slot = deploy_block_slot + 1) AS slot_plus_1_rows,
            count(DISTINCT token_address) FILTER (
                WHERE block_slot = deploy_block_slot + 1
            ) AS slot_plus_1_tokens,
            count(*) FILTER (WHERE block_slot = deploy_block_slot + 2) AS slot_plus_2_rows,
            count(DISTINCT token_address) FILTER (
                WHERE block_slot = deploy_block_slot + 2
            ) AS slot_plus_2_tokens,
            count(*) FILTER (WHERE block_slot > deploy_block_slot + 2) AS later_slot_rows,
            count(*) FILTER (WHERE block_time < deploy_block_time) AS pre_deploy_time_rows,
            count(*) FILTER (
                WHERE slot_index_id != printf('%012d%06d%04d', block_slot, tx_index, event_index)
            ) AS slot_index_encoding_mismatch_rows,
            count(*) FILTER (
                WHERE CAST(epoch(CAST(timestamp AS TIMESTAMPTZ)) AS BIGINT) != block_time
            ) AS timestamp_mismatch_rows,
            count(*) FILTER (
                WHERE creator_address != deploy_tx_signer
            ) AS creator_signer_mismatch_rows,
            count(*) FILTER (WHERE side NOT IN ('buy', 'sell')) AS invalid_side_rows,
            count(*) FILTER (WHERE program NOT IN ('pump', 'pump_amm')) AS invalid_program_rows,
            count(*) FILTER (
                WHERE price_usd <= 0 OR price_sol <= 0 OR base_amount <= 0
            ) AS nonpositive_price_or_base_rows,
            count(*) FILTER (
                WHERE amount_usd < 0 OR amount_sol < 0 OR quote_amount < 0
            ) AS negative_amount_rows,
            count(*) FILTER (
                WHERE quote_mint != 'So11111111111111111111111111111111111111112'
                   OR base_decimals != 6 OR quote_decimals != 9
            ) AS noncanonical_quote_rows,
            {", ".join(null_terms)}
        FROM {relation}
        """,
    )
    duplicate_summary = fetch_dict(
        connection,
        f"""
        SELECT
            count(*) AS duplicate_slot_index_groups,
            coalesce(sum(rows_in_group - 1), 0) AS duplicate_extra_rows
        FROM (
            SELECT count(*) AS rows_in_group
            FROM {relation}
            GROUP BY slot_index_id
            HAVING count(*) > 1
        )
        """,
    )
    deploy_context_summary = fetch_dict(
        connection,
        f"""
        SELECT count(*) AS tokens_with_multiple_deploy_contexts
        FROM (
            SELECT token_address
            FROM {relation}
            GROUP BY token_address
            HAVING count(DISTINCT struct_pack(
                deploy_block_time := deploy_block_time,
                deploy_block_slot := deploy_block_slot,
                deploy_tx_index := deploy_tx_index,
                deploy_tx_hash := deploy_tx_hash,
                deploy_tx_signer := deploy_tx_signer
            )) > 1
        )
        """,
    )
    connection.close()

    null_counts = {name: int(summary.pop(name)) for name in parquet.schema_arrow.names}
    quality_issues = []
    guarded_fields = {
        "pre_deploy_position_rows": int(summary["pre_deploy_position_rows"]),
        "pre_deploy_time_rows": int(summary["pre_deploy_time_rows"]),
        "slot_index_encoding_mismatch_rows": int(summary["slot_index_encoding_mismatch_rows"]),
        "timestamp_mismatch_rows": int(summary["timestamp_mismatch_rows"]),
        "invalid_side_rows": int(summary["invalid_side_rows"]),
        "invalid_program_rows": int(summary["invalid_program_rows"]),
        "nonpositive_price_or_base_rows": int(summary["nonpositive_price_or_base_rows"]),
        "negative_amount_rows": int(summary["negative_amount_rows"]),
        "noncanonical_quote_rows": int(summary["noncanonical_quote_rows"]),
        "duplicate_slot_index_groups": int(duplicate_summary["duplicate_slot_index_groups"]),
        "tokens_with_multiple_deploy_contexts": int(
            deploy_context_summary["tokens_with_multiple_deploy_contexts"]
        ),
    }
    expected_nullable = {"amount_usd", "amount_sol"}
    unexpected_nulls = {
        name: count
        for name, count in null_counts.items()
        if count and name not in expected_nullable
    }
    if unexpected_nulls:
        quality_issues.append("unexpected_null_values")
    if null_counts["amount_usd"] or null_counts["amount_sol"]:
        quality_issues.append("null_amount_fields_require_size_filter")
    quality_issues.extend(name for name, value in guarded_fields.items() if value)

    result = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "source": source.relative_to(PROJECT_ROOT).as_posix(),
        "source_role": "outcome_labels_and_backtest_only",
        "entry_feature_use_forbidden": True,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "rows": int(summary["rows"]),
        "expected_rows": EXPECTED_ROWS,
        "parquet_row_groups": parquet.metadata.num_row_groups,
        "unique_tokens": int(summary["unique_tokens"]),
        "null_counts": null_counts,
        "min_block_time": int(summary["min_block_time"]),
        "max_block_time": int(summary["max_block_time"]),
        "min_block_slot": int(summary["min_block_slot"]),
        "max_block_slot": int(summary["max_block_slot"]),
        **guarded_fields,
        "creator_signer_difference_rows": int(summary["creator_signer_mismatch_rows"]),
        "duplicate_extra_rows": int(duplicate_summary["duplicate_extra_rows"]),
        "same_deploy_transaction_rows": int(summary["same_deploy_transaction_rows"]),
        "executable_same_slot_rows": int(summary["executable_same_slot_rows"]),
        "executable_same_slot_tokens": int(summary["executable_same_slot_tokens"]),
        "slot_plus_1_rows": int(summary["slot_plus_1_rows"]),
        "slot_plus_1_tokens": int(summary["slot_plus_1_tokens"]),
        "slot_plus_2_rows": int(summary["slot_plus_2_rows"]),
        "slot_plus_2_tokens": int(summary["slot_plus_2_tokens"]),
        "later_slot_rows": int(summary["later_slot_rows"]),
        "required_replica_filters": [
            (
                "block_slot > deploy_block_slot OR "
                "(block_slot = deploy_block_slot AND tx_index > deploy_tx_index)"
            ),
            "price_usd > 0 AND price_sol > 0 AND base_amount > 0",
            "amount_usd >= 0 AND amount_sol >= 0 AND quote_amount >= 0",
            "amount_usd IS NOT NULL AND amount_sol IS NOT NULL for size or volume calculations",
        ],
        "exact_0_1_2_slot_delay_supported": True,
        "source_validation_status": "passed" if not quality_issues else "requires_explicit_filters",
        "quality_issues": quality_issues,
        "final_holdout_evaluated": False,
    }
    return result


def main() -> None:
    result = audit_source()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
