import hashlib
import io
import json
import struct
import urllib.request
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from solana_sniper.download_candles import USER_AGENT, parse_content_range
from solana_sniper.paths import PROJECT_ROOT, REPORT_DIR

CONFIG_PATH = PROJECT_ROOT / "config" / "competition.yaml"
OUTPUT_PATH = REPORT_DIR / "trades_remote_metadata_audit.json"
EXPECTED_BYTES = 17_944_584_022
EXPECTED_ROWS = 133_978_933
EXPECTED_SCHEMA = [
    ("token_address", "string"),
    ("slot_index_id", "string"),
    ("block_slot", "int64"),
    ("tx_index", "int32"),
    ("event_index", "int32"),
    ("tx_hash", "string"),
    ("timestamp", "string"),
    ("block_time", "int64"),
    ("user_address", "string"),
    ("side", "string"),
    ("program", "string"),
    ("price_usd", "double"),
    ("price_sol", "double"),
    ("amount_usd", "double"),
    ("amount_sol", "double"),
    ("base_amount", "double"),
    ("quote_amount", "double"),
    ("quote_mint", "string"),
    ("base_decimals", "int32"),
    ("quote_decimals", "int32"),
    ("deploy_block_time", "int64"),
    ("deploy_block_slot", "int64"),
    ("deploy_tx_index", "int32"),
    ("deploy_tx_hash", "string"),
    ("deploy_tx_signer", "string"),
    ("creator_address", "string"),
]


def schema_signature(schema: pa.Schema) -> list[tuple[str, str]]:
    return [(field.name, str(field.type)) for field in schema]


def validate_metadata(schema: pa.Schema, rows: int, remote_bytes: int) -> None:
    if schema_signature(schema) != EXPECTED_SCHEMA:
        raise ValueError(f"Unexpected trades schema: {schema_signature(schema)}")
    if rows != EXPECTED_ROWS:
        raise ValueError(f"Unexpected trades row count: {rows} != {EXPECTED_ROWS}")
    if remote_bytes != EXPECTED_BYTES:
        raise ValueError(f"Unexpected remote size: {remote_bytes} != {EXPECTED_BYTES}")


def supports_exact_slot_delay(schema: pa.Schema) -> bool:
    required = {
        "block_slot",
        "tx_index",
        "event_index",
        "deploy_block_slot",
        "deploy_tx_index",
        "price_usd",
        "price_sol",
    }
    return required.issubset(schema.names)


def fetch_range(url: str, start: int, end: int) -> tuple[bytes, dict[str, str | None]]:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content_range = parse_content_range(response.headers.get("Content-Range"))
        if response.status != 206 or content_range != (start, end, end + 1):
            raise OSError(
                f"Remote source did not honor exact Range {start}-{end}: "
                f"status={response.status}, content_range={content_range}"
            )
        return response.read(), {
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }


def json_value(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def audit_remote_metadata(url: str) -> dict[str, object]:
    head = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    with urllib.request.urlopen(head, timeout=30) as response:
        remote_bytes = int(response.headers["Content-Length"])
        etag = response.headers.get("ETag")
        last_modified = response.headers.get("Last-Modified")
        accept_ranges = response.headers.get("Accept-Ranges")

    footer, _ = fetch_range(url, remote_bytes - 8, remote_bytes - 1)
    if footer[4:] != b"PAR1":
        raise ValueError("Remote source does not end with Parquet magic")
    metadata_bytes = struct.unpack("<I", footer[:4])[0]
    metadata_start = remote_bytes - metadata_bytes - 8
    tail, range_headers = fetch_range(url, metadata_start, remote_bytes - 1)
    parquet = pq.ParquetFile(io.BytesIO(b"PAR1" + tail))
    validate_metadata(parquet.schema_arrow, parquet.metadata.num_rows, remote_bytes)

    statistics = {}
    for column_index, name in enumerate(parquet.schema_arrow.names):
        minima = []
        maxima = []
        groups_with_min_max = 0
        for group_index in range(parquet.metadata.num_row_groups):
            stats = parquet.metadata.row_group(group_index).column(column_index).statistics
            if stats is not None and stats.has_min_max:
                groups_with_min_max += 1
                minima.append(stats.min)
                maxima.append(stats.max)
        statistics[name] = {
            "row_groups_with_min_max": groups_with_min_max,
            "minimum": json_value(min(minima)) if minima else None,
            "maximum": json_value(max(maxima)) if maxima else None,
        }

    return {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "url": url,
        "remote_bytes": remote_bytes,
        "etag": etag,
        "last_modified": last_modified,
        "accept_ranges": accept_ranges,
        "range_etag": range_headers["etag"],
        "metadata_bytes": metadata_bytes,
        "metadata_range_start": metadata_start,
        "metadata_tail_sha256": hashlib.sha256(tail).hexdigest(),
        "parquet_rows": parquet.metadata.num_rows,
        "parquet_row_groups": parquet.metadata.num_row_groups,
        "created_by": parquet.metadata.created_by,
        "schema": [{"name": name, "type": kind} for name, kind in EXPECTED_SCHEMA],
        "column_statistics": statistics,
        "supports_exact_0_1_2_slot_delay": supports_exact_slot_delay(parquet.schema_arrow),
        "body_download_decision": "approved_by_schema_pending_full_file_validation",
        "source_role": "outcome_labels_and_backtest_only",
        "entry_feature_use_forbidden": True,
        "final_holdout_evaluated": False,
    }


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    urls = config["data_sources"]["optional_post_decision_sources"]
    matches = [url for url in urls if url.endswith("/pumpfun_trades.parquet")]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one trades source, found {len(matches)}")
    result = audit_remote_metadata(matches[0])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
