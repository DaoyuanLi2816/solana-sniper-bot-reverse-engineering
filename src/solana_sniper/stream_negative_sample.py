"""Stream a deterministic time-spanning negative sample without storing 14 GB raw JSON."""

import gzip
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from solana_sniper.extract_deploy_features import _instruction_features
from solana_sniper.manifest import sha256_file
from solana_sniper.paths import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR

CONFIG_PATH = PROJECT_ROOT / "config" / "competition.yaml"
NEGATIVE_RAW = "not_bought_deploy_txs.jsonl.gz"
NEGATIVE_INDEX = "not_bought_deploy_txs_index.parquet"


def _write_batch(writer, records: list[dict[str, object]]):
    frame = pd.DataFrame.from_records(records)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(_temporary_raw_path(), table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _temporary_raw_path() -> Path:
    return PROCESSED_DIR / "negative_raw_features.parquet.part"


def stream_sample(url: str, *, stride: int = 25, batch_size: int = 10_000) -> dict[str, int]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    index_destination = RAW_DIR / "core" / NEGATIVE_INDEX
    index_destination.parent.mkdir(parents=True, exist_ok=True)
    raw_final = PROCESSED_DIR / "negative_raw_features.parquet"
    raw_temporary = _temporary_raw_path()
    if raw_temporary.exists():
        raw_temporary.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": "solana-sniper-campaign/0.1"})
    sampled = 0
    total = 0
    writer = None
    found_raw = False
    found_index = False
    batch: list[dict[str, object]] = []
    with (
        urllib.request.urlopen(request, timeout=300) as response,
        tarfile.open(fileobj=response, mode="r|") as archive,
    ):
        for member in archive:
            name = Path(member.name).name
            if name == NEGATIVE_RAW and member.isfile():
                found_raw = True
                member_source = archive.extractfile(member)
                if member_source is None:
                    raise OSError("Could not open negative JSONL member")
                with gzip.GzipFile(fileobj=member_source, mode="rb") as compressed:
                    for total, line in enumerate(compressed, start=1):
                        if total % stride == 0:
                            row = json.loads(line)
                            batch.append({"line_number": total, **_instruction_features(row)})
                            sampled += 1
                            if len(batch) >= batch_size:
                                writer = _write_batch(writer, batch)
                                batch.clear()
                        if total % 250_000 == 0:
                            print(
                                json.dumps({"negative_rows_read": total, "sampled": sampled}),
                                flush=True,
                            )
                if batch:
                    writer = _write_batch(writer, batch)
                    batch.clear()
                if writer is not None:
                    writer.close()
                    writer = None
                raw_temporary.replace(raw_final)
            elif name == NEGATIVE_INDEX and member.isfile():
                found_index = True
                source = archive.extractfile(member)
                if source is None:
                    raise OSError("Could not open negative index member")
                temporary = index_destination.with_suffix(index_destination.suffix + ".part")
                with temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                if temporary.stat().st_size != member.size:
                    raise OSError("Incomplete negative index")
                temporary.replace(index_destination)
                break
    if not found_raw or not found_index:
        raise FileNotFoundError(
            f"Required archive members missing: raw={found_raw}, index={found_index}"
        )
    return {"negative_rows_read": total, "negative_rows_sampled": sampled, "stride": stride}


def join_index_and_features(stride: int) -> Path:
    raw_features = PROCESSED_DIR / "negative_raw_features.parquet"
    index = RAW_DIR / "core" / NEGATIVE_INDEX
    output = PROCESSED_DIR / "negative_deploy_features.parquet"
    temporary = output.with_suffix(output.suffix + ".part")
    connection = duckdb.connect()
    connection.execute(
        """
        COPY (
            SELECT
                i.tx_hash,
                i.line_number,
                i.blockTime,
                i.blockSlot,
                i.token_address,
                i.tx_signer,
                i.creator_address,
                to_timestamp(f.decision_time) AS decision_time,
                f.block_slot,
                f.transaction_index,
                f.tx_fee_lamports,
                f.priority_fee_lamports,
                f.compute_units,
                f.cost_units,
                f.outer_instruction_count,
                f.inner_instruction_count,
                f.account_count,
                f.signature_count,
                f.log_message_count,
                f.pre_token_balance_count,
                f.post_token_balance_count,
                f.signer_lamport_delta,
                f.metadata_name_length,
                f.metadata_symbol_length,
                f.metadata_uri_length,
                f.metadata_uri_is_ipfs,
                f.metadata_present,
                f.transaction_error,
                0 AS label,
                hour(to_timestamp(f.decision_time)) AS decision_hour_utc,
                dayofweek(to_timestamp(f.decision_time)) AS decision_weekday_utc
            FROM read_parquet(?) AS f
            INNER JOIN read_parquet(?) AS i USING (line_number)
            WHERE i.line_number % ? = 0
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(raw_features), str(index), stride, str(temporary)],
    )
    connection.close()
    temporary.replace(output)
    return output


def combine_classification_dataset() -> Path:
    positive = PROCESSED_DIR / "positive_deploy_features.parquet"
    negative = PROCESSED_DIR / "negative_deploy_features.parquet"
    output = PROCESSED_DIR / "classification_dataset.parquet"
    temporary = output.with_suffix(output.suffix + ".part")
    connection = duckdb.connect()
    connection.execute(
        "COPY (SELECT * FROM read_parquet(?) UNION ALL BY NAME SELECT * FROM read_parquet(?)) "
        "TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
        [str(positive), str(negative), str(temporary)],
    )
    connection.close()
    temporary.replace(output)
    return output


def main() -> None:
    stride = 25
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    url = config["data_sources"]["core_archive_url"]
    raw_features = PROCESSED_DIR / "negative_raw_features.parquet"
    index = RAW_DIR / "core" / NEGATIVE_INDEX
    if raw_features.exists() and index.exists():
        status = {
            "negative_rows_read": 5_059_880,
            "negative_rows_sampled": pq.ParquetFile(raw_features).metadata.num_rows,
            "stride": stride,
            "reused": True,
        }
    else:
        status = stream_sample(url, stride=stride)
    negative = join_index_and_features(stride)
    combined = combine_classification_dataset()
    result = {
        **status,
        "negative_features": str(negative),
        "negative_features_sha256": sha256_file(negative),
        "classification_dataset": str(combined),
        "classification_dataset_sha256": sha256_file(combined),
    }
    manifest = PROCESSED_DIR / "negative_sample_manifest.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
