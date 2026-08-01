import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from solana_sniper.manifest import sha256_file
from solana_sniper.paths import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, REPORT_DIR

CLASSIFICATION_PATH = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
TRADES_PATH = RAW_DIR / "june" / "pumpfun_trades.parquet"
TRADES_AUDIT_PATH = REPORT_DIR / "trades_source_audit.json"
OUTPUT_PATH = PROCESSED_DIR / "replica_entry_prices.parquet"
METRICS_PATH = REPORT_DIR / "entry_prices_metrics.json"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "entry_price_coverage.svg"
JUNE_START = 1_780_272_000
JUNE_END = 1_782_863_999
DELAYS = (0, 1, 2)
EXPECTED_COLUMNS = [
    "token_address",
    "decision_time",
    "deploy_block_time",
    "deploy_block_slot",
    "deploy_tx_index",
    "requested_delay_slots",
    "target_entry_slot",
    "covered",
    "entry_slot_index_id",
    "entry_block_slot",
    "entry_tx_index",
    "entry_event_index",
    "entry_block_time",
    "entry_timestamp",
    "entry_tx_hash",
    "entry_side",
    "entry_program",
    "entry_price_usd",
    "entry_price_sol",
    "actual_delay_slots",
    "wait_slots_beyond_target",
]


def relation(path: Path) -> str:
    escaped = path.as_posix().replace("'", "''")
    return f"read_parquet('{escaped}')"


def selected_entry_sql(universe_relation: str, trades_relation: str) -> str:
    return f"""
        WITH delays(requested_delay_slots) AS (VALUES (0), (1), (2)),
        first_ids AS (
            SELECT
                u.token_address,
                d.requested_delay_slots,
                min(t.slot_index_id) AS entry_slot_index_id
            FROM {universe_relation} AS u
            CROSS JOIN delays AS d
            LEFT JOIN {trades_relation} AS t
              ON t.token_address = u.token_address
             AND t.block_slot >= u.deploy_block_slot + d.requested_delay_slots
            GROUP BY u.token_address, d.requested_delay_slots
        )
        SELECT
            u.token_address,
            u.decision_time,
            u.deploy_block_time,
            u.deploy_block_slot,
            u.deploy_tx_index,
            f.requested_delay_slots,
            u.deploy_block_slot + f.requested_delay_slots AS target_entry_slot,
            t.slot_index_id IS NOT NULL AS covered,
            t.slot_index_id AS entry_slot_index_id,
            t.block_slot AS entry_block_slot,
            t.tx_index AS entry_tx_index,
            t.event_index AS entry_event_index,
            t.block_time AS entry_block_time,
            t.timestamp AS entry_timestamp,
            t.tx_hash AS entry_tx_hash,
            t.side AS entry_side,
            t.program AS entry_program,
            t.price_usd AS entry_price_usd,
            t.price_sol AS entry_price_sol,
            t.block_slot - u.deploy_block_slot AS actual_delay_slots,
            t.block_slot - (u.deploy_block_slot + f.requested_delay_slots)
                AS wait_slots_beyond_target
        FROM first_ids AS f
        JOIN {universe_relation} AS u USING (token_address)
        LEFT JOIN {trades_relation} AS t
          ON t.slot_index_id = f.entry_slot_index_id
        ORDER BY u.decision_time, u.token_address, f.requested_delay_slots
    """


def validate_entry_prices(frame: pd.DataFrame, expected_tokens: int) -> None:
    if frame.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected entry-price columns: {frame.columns.tolist()}")
    if len(frame) != expected_tokens * len(DELAYS):
        raise ValueError(f"Unexpected entry-price rows: {len(frame)}")
    if frame.duplicated(["token_address", "requested_delay_slots"]).any():
        raise ValueError("Duplicate token-delay entry-price keys")
    delay_counts = frame.groupby("token_address")["requested_delay_slots"].agg(
        lambda values: tuple(sorted(values.tolist()))
    )
    if not delay_counts.map(lambda values: values == DELAYS).all():
        raise ValueError("Each token must have exactly the 0/1/2 delay rows")
    if not (
        frame["target_entry_slot"] == frame["deploy_block_slot"] + frame["requested_delay_slots"]
    ).all():
        raise ValueError("Target entry slot does not match requested delay")

    covered = frame[frame["covered"]]
    if (
        covered[
            [
                "entry_slot_index_id",
                "entry_block_slot",
                "entry_tx_index",
                "entry_event_index",
                "entry_block_time",
                "entry_price_usd",
                "entry_price_sol",
            ]
        ]
        .isna()
        .any(axis=None)
    ):
        raise ValueError("Covered entry rows contain null execution fields")
    if (covered["entry_block_slot"] < covered["target_entry_slot"]).any():
        raise ValueError("Entry occurs before the requested target slot")
    if (
        (covered["entry_block_slot"] == covered["deploy_block_slot"])
        & (covered["entry_tx_index"] <= covered["deploy_tx_index"])
    ).any():
        raise ValueError("Entry occurs in or before the deployment transaction")
    if (covered[["entry_price_usd", "entry_price_sol"]] <= 0).any(axis=None):
        raise ValueError("Covered entry rows contain nonpositive prices")
    if not (
        covered["actual_delay_slots"] == covered["entry_block_slot"] - covered["deploy_block_slot"]
    ).all():
        raise ValueError("Actual delay calculation mismatch")
    if not (
        covered["wait_slots_beyond_target"]
        == covered["actual_delay_slots"] - covered["requested_delay_slots"]
    ).all():
        raise ValueError("Wait-slot calculation mismatch")

    missing = frame[~frame["covered"]]
    execution_columns = [column for column in EXPECTED_COLUMNS if column.startswith("entry_")]
    execution_columns += ["actual_delay_slots", "wait_slots_beyond_target"]
    if missing[execution_columns].notna().any(axis=None):
        raise ValueError("Missing entry rows contain populated execution fields")

    ordered = frame.sort_values(["token_address", "requested_delay_slots"])
    positions = ordered.groupby("token_address")["entry_slot_index_id"].apply(list)
    for token_positions in positions:
        present = [position for position in token_positions if pd.notna(position)]
        if present != sorted(present):
            raise ValueError("Entry positions are not monotonic across delays")


def fetch_dict(connection: duckdb.DuckDBPyConnection, query: str) -> dict[str, object]:
    cursor = connection.execute(query)
    row = cursor.fetchone()
    if row is None:
        raise ValueError("Entry-price query unexpectedly returned no rows")
    return dict(zip((column[0] for column in cursor.description), row, strict=True))


def write_coverage_svg(coverage: list[dict[str, object]], path: Path) -> None:
    width, height = 760, 420
    chart_top, chart_bottom = 72, 330
    chart_height = chart_bottom - chart_top
    max_rows = max(int(item["rows"]) for item in coverage)
    colors = {"exact": "#2563eb", "later": "#f59e0b", "missing": "#94a3b8"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="34" font-family="Arial" font-size="22" font-weight="700" '
        'fill="#0f172a">0/1/2-slot entry proxy coverage</text>',
        '<text x="36" y="56" font-family="Arial" font-size="13" fill="#475569">'
        "Exact target slot, later observed trade, and no later trade</text>",
    ]
    for index, item in enumerate(coverage):
        x = 120 + index * 200
        y = chart_bottom
        segments = [
            ("exact", int(item["exact_target_slot_rows"])),
            ("later", int(item["later_than_target_rows"])),
            ("missing", int(item["missing_rows"])),
        ]
        for name, rows in segments:
            segment_height = chart_height * rows / max_rows
            y -= segment_height
            elements.append(
                f'<rect x="{x}" y="{y:.2f}" width="90" height="{segment_height:.2f}" '
                f'fill="{colors[name]}"/>'
            )
        delay = int(item["requested_delay_slots"])
        coverage_rate = 100 * float(item["coverage_rate"])
        elements.extend(
            [
                f'<text x="{x + 45}" y="356" text-anchor="middle" font-family="Arial" '
                f'font-size="15" fill="#0f172a">delay {delay}</text>',
                f'<text x="{x + 45}" y="378" text-anchor="middle" font-family="Arial" '
                f'font-size="13" fill="#475569">{coverage_rate:.2f}% covered</text>',
            ]
        )
    legend = [
        ("exact", "Exact target slot"),
        ("later", "Later observed trade"),
        ("missing", "Missing"),
    ]
    for index, (name, label) in enumerate(legend):
        x = 110 + index * 190
        elements.append(f'<rect x="{x}" y="397" width="14" height="14" fill="{colors[name]}"/>')
        elements.append(
            f'<text x="{x + 20}" y="409" font-family="Arial" font-size="12" '
            f'fill="#334155">{label}</text>'
        )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    for path in (CLASSIFICATION_PATH, TRADES_PATH, TRADES_AUDIT_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    source_audit = json.loads(TRADES_AUDIT_PATH.read_text(encoding="utf-8"))
    if not source_audit["exact_0_1_2_slot_delay_supported"]:
        raise ValueError("Trades source is not approved for exact slot-delay construction")
    if source_audit["bytes"] != TRADES_PATH.stat().st_size:
        raise ValueError("Trades file size differs from its approved audit")

    temporary_dir = PROCESSED_DIR / "duckdb_entry_prices_tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '12GB'")
    connection.execute("SET preserve_insertion_order = false")
    temp_sql_path = temporary_dir.as_posix().replace("'", "''")
    connection.execute(f"SET temp_directory = '{temp_sql_path}'")

    classification = relation(CLASSIFICATION_PATH)
    trades = relation(TRADES_PATH)
    connection.execute(
        f"""
        CREATE TEMP TABLE universe AS
        SELECT
            token_address,
            decision_time,
            blockTime AS deploy_block_time,
            blockSlot AS deploy_block_slot,
            transaction_index AS deploy_tx_index
        FROM {classification}
        WHERE blockTime BETWEEN {JUNE_START} AND {JUNE_END}
        """
    )
    universe_audit = fetch_dict(
        connection,
        """
        SELECT
            count(*) AS rows,
            count(DISTINCT token_address) AS unique_tokens,
            min(deploy_block_time) AS min_deploy_block_time,
            max(deploy_block_time) AS max_deploy_block_time
        FROM universe
        """,
    )
    if universe_audit["rows"] != universe_audit["unique_tokens"]:
        raise ValueError("June classification universe contains duplicate token addresses")

    context_audit = fetch_dict(
        connection,
        f"""
        WITH contexts AS (
            SELECT
                token_address,
                min(deploy_block_time) AS deploy_block_time,
                min(deploy_block_slot) AS deploy_block_slot,
                min(deploy_tx_index) AS deploy_tx_index
            FROM {trades}
            GROUP BY token_address
        )
        SELECT
            count(contexts.token_address) AS matched_tokens,
            count(*) FILTER (WHERE contexts.token_address IS NULL) AS tokens_without_trades,
            count(*) FILTER (
                WHERE contexts.token_address IS NOT NULL
                  AND universe.deploy_block_time != contexts.deploy_block_time
            ) AS deploy_time_mismatches,
            count(*) FILTER (
                WHERE contexts.token_address IS NOT NULL
                  AND universe.deploy_block_slot != contexts.deploy_block_slot
            ) AS deploy_slot_mismatches,
            count(*) FILTER (
                WHERE contexts.token_address IS NOT NULL
                  AND universe.deploy_tx_index != contexts.deploy_tx_index
            ) AS deploy_tx_index_mismatches
        FROM universe
        LEFT JOIN contexts USING (token_address)
        """,
    )
    if any(
        context_audit[name]
        for name in (
            "deploy_time_mismatches",
            "deploy_slot_mismatches",
            "deploy_tx_index_mismatches",
        )
    ):
        raise ValueError(f"Classification/trades deployment context mismatch: {context_audit}")

    connection.execute(
        f"""
        CREATE TEMP TABLE relevant_trades AS
        SELECT
            t.token_address,
            t.slot_index_id,
            t.block_slot,
            t.tx_index,
            t.event_index,
            t.block_time,
            t.timestamp,
            t.tx_hash,
            t.side,
            t.program,
            t.price_usd,
            t.price_sol
        FROM {trades} AS t
        JOIN universe AS u USING (token_address)
        WHERE t.block_slot > u.deploy_block_slot
           OR (t.block_slot = u.deploy_block_slot AND t.tx_index > u.deploy_tx_index)
        """
    )

    temporary_output = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".part")
    if temporary_output.exists():
        raise FileExistsError(f"Refusing to overwrite active partial output: {temporary_output}")
    output_sql_path = temporary_output.as_posix().replace("'", "''")
    select_sql = selected_entry_sql("universe", "relevant_trades")
    connection.execute(
        f"COPY ({select_sql}) TO '{output_sql_path}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    connection.close()

    frame = pq.read_table(temporary_output).to_pandas()
    validate_entry_prices(frame, int(universe_audit["rows"]))
    temporary_output.replace(OUTPUT_PATH)
    output_hash = sha256_file(OUTPUT_PATH)

    coverage = []
    for delay, group in frame.groupby("requested_delay_slots", sort=True):
        covered = group[group["covered"]]
        coverage.append(
            {
                "requested_delay_slots": int(delay),
                "rows": len(group),
                "covered_rows": len(covered),
                "missing_rows": int((~group["covered"]).sum()),
                "coverage_rate": float(group["covered"].mean()),
                "exact_target_slot_rows": int(
                    (covered["entry_block_slot"] == covered["target_entry_slot"]).sum()
                ),
                "later_than_target_rows": int(
                    (covered["entry_block_slot"] > covered["target_entry_slot"]).sum()
                ),
                "median_wait_slots_beyond_target": float(
                    covered["wait_slots_beyond_target"].median()
                ),
                "p95_wait_slots_beyond_target": float(
                    covered["wait_slots_beyond_target"].quantile(0.95)
                ),
            }
        )

    write_coverage_svg(coverage, FIGURE_PATH)
    metrics = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "hypothesis": "first_observed_trade_at_or_after_target_slot_is_deterministic",
        "entry_definition": (
            "first trade ordered by slot_index_id at or after deploy_block_slot + requested delay, "
            "excluding the deployment transaction"
        ),
        "source_role": "outcome_and_backtest_only",
        "entry_feature_use_forbidden": True,
        "classification_path": CLASSIFICATION_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "classification_sha256": sha256_file(CLASSIFICATION_PATH),
        "trades_path": TRADES_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "trades_sha256": source_audit["sha256"],
        "output_path": OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "output_sha256": output_hash,
        "rows": len(frame),
        "unique_tokens": int(frame["token_address"].nunique()),
        "unique_token_delay_keys": int(
            frame[["token_address", "requested_delay_slots"]].drop_duplicates().shape[0]
        ),
        "universe_audit": universe_audit,
        "context_audit": context_audit,
        "coverage": coverage,
        "figure_path": FIGURE_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "figure_sha256": sha256_file(FIGURE_PATH),
        "final_holdout_evaluated": False,
        "predictions_or_returns_generated": False,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
