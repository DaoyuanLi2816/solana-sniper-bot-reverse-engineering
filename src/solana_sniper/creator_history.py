"""Build strict pre-decision deployer-frequency features from deployment indexes."""

import json
from pathlib import Path

import duckdb

from solana_sniper.manifest import sha256_file
from solana_sniper.paths import PROCESSED_DIR, RAW_DIR, project_relative


def build_creator_history_dataset(
    classification_path: Path,
    positive_index_path: Path,
    negative_index_path: Path,
    output_path: Path,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    connection = duckdb.connect()
    connection.read_parquet(str(classification_path)).create_view("classification_features")
    connection.read_parquet(str(positive_index_path)).create_view("positive_index")
    connection.read_parquet(str(negative_index_path)).create_view("negative_index")
    connection.execute(
        """
        CREATE TEMP TABLE creator_slot_history AS
        WITH universe AS (
            SELECT tx_signer, blockSlot, blockTime, token_address FROM positive_index
            UNION ALL
            SELECT tx_signer, blockSlot, blockTime, token_address FROM negative_index
        ),
        creator_slots AS (
            SELECT
                tx_signer,
                blockSlot,
                min(blockTime) AS block_time,
                count(*) AS deployments_in_slot
            FROM universe
            GROUP BY tx_signer, blockSlot
        )
        SELECT
            tx_signer,
            blockSlot,
            coalesce(
                sum(deployments_in_slot) OVER (
                    PARTITION BY tx_signer ORDER BY blockSlot
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ),
                0
            )::BIGINT AS creator_prior_deploy_count,
            coalesce(
                count(*) OVER (
                    PARTITION BY tx_signer ORDER BY blockSlot
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ),
                0
            )::BIGINT AS creator_prior_active_slot_count,
            min(block_time) OVER (
                PARTITION BY tx_signer ORDER BY blockSlot
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS creator_first_prior_deploy_time,
            min(blockSlot) OVER (
                PARTITION BY tx_signer ORDER BY blockSlot
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS creator_first_prior_deploy_slot,
            lag(block_time) OVER (
                PARTITION BY tx_signer ORDER BY blockSlot
            ) AS creator_previous_deploy_time,
            lag(blockSlot) OVER (
                PARTITION BY tx_signer ORDER BY blockSlot
            ) AS creator_previous_deploy_slot,
            block_time
        FROM creator_slots
        """
    )
    violation_count = connection.execute(
        """
        SELECT count(*)
        FROM creator_slot_history
        WHERE creator_previous_deploy_slot >= blockSlot
           OR creator_first_prior_deploy_slot >= blockSlot
        """
    ).fetchone()[0]
    if violation_count:
        raise ValueError(f"Found {violation_count} non-historical creator feature rows")
    connection.execute(
        """
        COPY (
            SELECT
                c.*,
                h.creator_prior_deploy_count,
                h.creator_prior_active_slot_count,
                CASE
                    WHEN h.creator_first_prior_deploy_time IS NULL THEN NULL
                    ELSE epoch(c.decision_time)::BIGINT - h.creator_first_prior_deploy_time
                END AS creator_observed_age_seconds,
                CASE
                    WHEN h.creator_previous_deploy_time IS NULL THEN NULL
                    ELSE epoch(c.decision_time)::BIGINT - h.creator_previous_deploy_time
                END AS creator_seconds_since_previous_deploy
            FROM classification_features AS c
            INNER JOIN creator_slot_history AS h
              ON c.tx_signer = h.tx_signer AND c.blockSlot = h.blockSlot
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(temporary)],
    )
    source_stats = connection.execute(
        "SELECT count(*), count(distinct token_address) FROM classification_features"
    ).fetchone()
    output_stats = connection.execute(
        """
        SELECT
            count(*),
            count(distinct token_address),
            min(creator_prior_deploy_count),
            min(creator_prior_active_slot_count),
            sum(creator_observed_age_seconds < 0),
            sum(creator_seconds_since_previous_deploy < 0)
        FROM read_parquet(?)
        """,
        [str(temporary)],
    ).fetchone()
    connection.close()
    if output_stats[0] != source_stats[0] or output_stats[1] != source_stats[1]:
        raise ValueError(
            f"Creator-history join changed dataset cardinality: {source_stats} -> {output_stats}"
        )
    if output_stats[2] < 0 or output_stats[3] < 0 or output_stats[4] or output_stats[5]:
        raise ValueError(f"Invalid creator-history values: {output_stats}")
    temporary.replace(output_path)
    try:
        reported_path = project_relative(output_path)
    except ValueError:
        reported_path = output_path.as_posix()
    return {
        "path": reported_path,
        "sha256": sha256_file(output_path),
        "rows": int(output_stats[0]),
        "unique_tokens": int(output_stats[1]),
        "strict_time_violations": int(violation_count),
        "feature_family": "creator_prior_deployment_frequency_and_recency",
        "creator_key": "deployment_index.tx_signer",
    }


def main() -> None:
    result = build_creator_history_dataset(
        PROCESSED_DIR / "classification_dataset.parquet",
        RAW_DIR / "core" / "bought_deploy_txs_index.parquet",
        RAW_DIR / "core" / "not_bought_deploy_txs_index.parquet",
        PROCESSED_DIR / "classification_dataset_creator_history.parquet",
    )
    manifest_path = PROCESSED_DIR / "creator_history_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
