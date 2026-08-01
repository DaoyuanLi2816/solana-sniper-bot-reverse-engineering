import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from solana_sniper.manifest import sha256_file
from solana_sniper.paths import PROJECT_ROOT, RAW_DIR, REPORT_DIR

SOURCE_PATH = RAW_DIR / "june" / "mcap_candles.parquet"
OUTPUT_PATH = REPORT_DIR / "candles_source_audit.json"
EXPECTED_ROWS = 60_109_034
EXPECTED_SCHEMA = [
    ("token_address", "string"),
    ("resolution", "string"),
    ("deploy_time_ms", "int64"),
    ("deploy_time_s", "int64"),
    ("candle_time_ms", "int64"),
    ("candle_time_s", "int64"),
    ("open_mcap", "double"),
    ("high_mcap", "double"),
    ("low_mcap", "double"),
    ("close_mcap", "double"),
    ("volume", "double"),
    ("amount", "double"),
]


def schema_signature(schema: pa.Schema) -> list[tuple[str, str]]:
    return [(field.name, str(field.type)) for field in schema]


def validate_source_metadata(schema: pa.Schema, rows: int) -> None:
    actual_schema = schema_signature(schema)
    if actual_schema != EXPECTED_SCHEMA:
        raise ValueError(f"Unexpected candle schema: {actual_schema}")
    if rows != EXPECTED_ROWS:
        raise ValueError(f"Unexpected candle row count: {rows} != {EXPECTED_ROWS}")


def supports_exact_slot_delay(schema: pa.Schema) -> bool:
    names = set(schema.names)
    return "block_slot" in names and "transaction_index" in names


def milliseconds_to_iso(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000, tz=UTC).isoformat()


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
    validate_source_metadata(parquet.schema_arrow, parquet.metadata.num_rows)

    temporary = PROJECT_ROOT / "data" / "processed" / "duckdb_candles_audit_tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    parquet_sql_path = source.as_posix().replace("'", "''")
    relation = f"read_parquet('{parquet_sql_path}')"
    null_terms = [f'count(*) - count("{name}") AS "{name}"' for name, _ in EXPECTED_SCHEMA]

    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '8GB'")
    temp_sql_path = temporary.as_posix().replace("'", "''")
    connection.execute(f"SET temp_directory = '{temp_sql_path}'")
    summary = fetch_dict(
        connection,
        f"""
        SELECT
            count(*) AS rows,
            count(DISTINCT token_address) AS unique_tokens,
            min(deploy_time_ms) AS min_deploy_time_ms,
            max(deploy_time_ms) AS max_deploy_time_ms,
            min(candle_time_ms) AS min_candle_time_ms,
            max(candle_time_ms) AS max_candle_time_ms,
            count(*) FILTER (WHERE candle_time_ms < deploy_time_ms) AS pre_deploy_rows,
            count(DISTINCT token_address) FILTER (
                WHERE candle_time_ms < deploy_time_ms
            ) AS pre_deploy_tokens,
            min(candle_time_ms - deploy_time_ms) FILTER (
                WHERE candle_time_ms < deploy_time_ms
            ) AS minimum_pre_deploy_delta_ms,
            max(candle_time_ms - deploy_time_ms) FILTER (
                WHERE candle_time_ms < deploy_time_ms
            ) AS maximum_pre_deploy_delta_ms,
            count(*) FILTER (
                WHERE deploy_time_ms != deploy_time_s * 1000
                   OR candle_time_ms != candle_time_s * 1000
            ) AS second_alignment_violations,
            count(*) FILTER (
                WHERE low_mcap > high_mcap
                   OR open_mcap < low_mcap OR open_mcap > high_mcap
                   OR close_mcap < low_mcap OR close_mcap > high_mcap
            ) AS invalid_ohlc_rows,
            count(*) FILTER (WHERE volume < 0 OR amount < 0) AS negative_flow_rows,
            {", ".join(null_terms)}
        FROM {relation}
        """,
    )
    resolutions = [
        {"resolution": row[0], "rows": row[1]}
        for row in connection.execute(
            f"SELECT resolution, count(*) FROM {relation} GROUP BY resolution ORDER BY resolution"
        ).fetchall()
    ]
    duplicate_summary = fetch_dict(
        connection,
        f"""
        SELECT
            count(*) AS duplicate_key_groups,
            coalesce(sum(rows_in_group - 1), 0) AS duplicate_extra_rows
        FROM (
            SELECT count(*) AS rows_in_group
            FROM {relation}
            GROUP BY token_address, resolution, candle_time_ms
            HAVING count(*) > 1
        )
        """,
    )
    connection.close()

    null_counts = {name: int(summary.pop(name)) for name, _ in EXPECTED_SCHEMA}
    temporal = {
        "min_deploy_time_ms": int(summary.pop("min_deploy_time_ms")),
        "max_deploy_time_ms": int(summary.pop("max_deploy_time_ms")),
        "min_candle_time_ms": int(summary.pop("min_candle_time_ms")),
        "max_candle_time_ms": int(summary.pop("max_candle_time_ms")),
    }
    quality_issues = []
    if any(null_counts.values()):
        quality_issues.append("null_values")
    if summary["pre_deploy_rows"]:
        quality_issues.append("pre_deploy_candles_require_exclusion")
    if summary["second_alignment_violations"]:
        quality_issues.append("second_timestamp_alignment")
    if summary["invalid_ohlc_rows"]:
        quality_issues.append("invalid_ohlc")
    if summary["negative_flow_rows"]:
        quality_issues.append("negative_volume_or_amount")
    if duplicate_summary["duplicate_key_groups"]:
        quality_issues.append("duplicate_token_resolution_time_keys")

    result = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "source": source.relative_to(PROJECT_ROOT).as_posix(),
        "source_role": "outcome_labels_and_backtest_only",
        "entry_feature_use_forbidden": True,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "schema": [{"name": name, "type": kind} for name, kind in EXPECTED_SCHEMA],
        "parquet_row_groups": parquet.metadata.num_row_groups,
        "rows": int(summary["rows"]),
        "unique_tokens": int(summary["unique_tokens"]),
        "resolutions": resolutions,
        "null_counts": null_counts,
        "temporal_bounds": {
            **temporal,
            "min_deploy_time_utc": milliseconds_to_iso(temporal["min_deploy_time_ms"]),
            "max_deploy_time_utc": milliseconds_to_iso(temporal["max_deploy_time_ms"]),
            "min_candle_time_utc": milliseconds_to_iso(temporal["min_candle_time_ms"]),
            "max_candle_time_utc": milliseconds_to_iso(temporal["max_candle_time_ms"]),
        },
        "pre_deploy_rows": int(summary["pre_deploy_rows"]),
        "pre_deploy_tokens": int(summary["pre_deploy_tokens"]),
        "minimum_pre_deploy_delta_ms": int(summary["minimum_pre_deploy_delta_ms"]),
        "maximum_pre_deploy_delta_ms": int(summary["maximum_pre_deploy_delta_ms"]),
        "required_outcome_filters": [
            "candle_time_ms >= deploy_time_ms",
            (
                "low_mcap <= high_mcap AND open_mcap BETWEEN low_mcap AND high_mcap "
                "AND close_mcap BETWEEN low_mcap AND high_mcap"
            ),
        ],
        "second_alignment_violations": int(summary["second_alignment_violations"]),
        "invalid_ohlc_rows": int(summary["invalid_ohlc_rows"]),
        "negative_flow_rows": int(summary["negative_flow_rows"]),
        "duplicate_key_groups": int(duplicate_summary["duplicate_key_groups"]),
        "duplicate_extra_rows": int(duplicate_summary["duplicate_extra_rows"]),
        "exact_0_1_2_slot_delay_supported": supports_exact_slot_delay(parquet.schema_arrow),
        "slot_delay_limitation": (
            "Per-second candles contain neither block_slot nor transaction_index; they cannot "
            "establish exact 0/1/2-slot entry prices."
        ),
        "source_validation_status": "passed" if not quality_issues else "requires_explicit_filters",
        "quality_issues": quality_issues,
        "outcome_label_source_accepted_with_filters": True,
        "exact_slot_backtest_source_accepted": False,
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
