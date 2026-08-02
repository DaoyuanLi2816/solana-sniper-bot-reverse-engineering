"""Audit zero-slot replica feasibility with a training-derived transaction-position lag."""

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from solana_sniper.baseline import (
    NON_FEATURE_COLUMNS,
    _best_f1_threshold,
    _fixed_threshold_metrics,
    _population_weights,
)
from solana_sniper.boosting import BOOSTING_PARAMETERS, build_boosting_model
from solana_sniper.guardrails import assert_feature_names_are_pre_decision
from solana_sniper.manifest import append_experiment, git_head, sha256_file
from solana_sniper.paths import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, REPORT_DIR, project_relative
from solana_sniper.replica_backtest import (
    HOLD_SECONDS,
    JUNE_START,
    build_exit_marks,
    calculate_portfolio_metrics,
    training_behavior_parameters,
    validate_backtest_rows,
)
from solana_sniper.splits import chronological_train_validation_test_split

CLASSIFICATION_PATH = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
ENTRY_LATENCY_PATH = PROCESSED_DIR / "entry_latency.parquet"
ENTRY_PRICES_PATH = PROCESSED_DIR / "replica_entry_prices.parquet"
WALLET_PATH = RAW_DIR / "wallet" / "5brv79e_activity.parquet"
TRADES_PATH = RAW_DIR / "june" / "pumpfun_trades.parquet"
PRIOR_BACKTEST_PATH = REPORT_DIR / "replica_validation_backtest.json"
METRICS_PATH = REPORT_DIR / "position_lag_validation_backtest.json"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "position_lag_validation_backtest.svg"


def training_position_parameters(
    latency: pd.DataFrame, train_end: pd.Timestamp
) -> dict[str, object]:
    """Summarize target-wallet same-slot transaction positions through train_end only."""
    positions = latency.loc[
        (latency["decision_time"] <= train_end)
        & (latency["latency_slots"] == 0)
        & latency["same_slot_position_delta"].notna(),
        "same_slot_position_delta",
    ]
    positions = pd.to_numeric(positions, errors="coerce").dropna()
    positions = positions.loc[positions > 0]
    if positions.empty:
        raise ValueError("No positive training-period same-slot position deltas")
    return {
        "cutoff_utc": train_end.isoformat(),
        "rows": int(len(positions)),
        "minimum": int(positions.min()),
        "p10": float(positions.quantile(0.1)),
        "p25": float(positions.quantile(0.25)),
        "median": float(positions.median()),
        "p75": float(positions.quantile(0.75)),
        "p90": float(positions.quantile(0.9)),
        "maximum": int(positions.max()),
        "share_at_or_before_10": float((positions <= 10).mean()),
        "share_at_or_before_50": float((positions <= 50).mean()),
        "share_at_or_before_100": float((positions <= 100).mean()),
    }


def _relation(path: Path) -> str:
    escaped = path.as_posix().replace("'", "''")
    return f"read_parquet('{escaped}')"


def build_position_entries(
    candidates: pd.DataFrame,
    trades_path: Path,
    *,
    position_lag: int,
    outcome_cutoff_epoch: int,
) -> pd.DataFrame:
    """Choose the first same-slot trade at or after a fixed transaction-position lag."""
    if candidates.empty or candidates["token_address"].duplicated().any():
        raise ValueError("Candidates must be nonempty and unique by token")
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '12GB'")
    connection.register("candidate_frame", candidates)
    connection.execute("CREATE TEMP TABLE candidates AS SELECT * FROM candidate_frame")
    trades = _relation(trades_path)
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
        SEMI JOIN candidates AS c ON t.token_address = c.token_address
        WHERE t.block_time < {outcome_cutoff_epoch}
        """
    )
    entry_ids = connection.execute(
        f"""
        SELECT
            c.token_address,
            min(t.slot_index_id) AS entry_slot_index_id
        FROM candidates AS c
        LEFT JOIN relevant_trades AS t
          ON t.token_address = c.token_address
         AND t.block_slot = c.deploy_block_slot
         AND t.tx_index >= c.deploy_tx_index + {position_lag}
        GROUP BY c.token_address
        """
    ).fetchdf()
    connection.register("entry_ids_frame", entry_ids)
    entries = connection.execute(
        """
        SELECT
            c.*,
            0 AS requested_delay_slots,
            e.entry_slot_index_id IS NOT NULL AS covered,
            e.entry_slot_index_id,
            t.block_slot AS entry_block_slot,
            t.tx_index AS entry_tx_index,
            t.event_index AS entry_event_index,
            t.block_time AS entry_block_time,
            t.timestamp AS entry_timestamp,
            t.tx_hash AS entry_tx_hash,
            t.side AS entry_side,
            t.program AS entry_program,
            t.price_usd AS entry_price_usd,
            t.price_sol AS entry_price_sol
        FROM candidates AS c
        JOIN entry_ids_frame AS e USING (token_address)
        LEFT JOIN relevant_trades AS t ON t.slot_index_id = e.entry_slot_index_id
        ORDER BY c.decision_time, c.token_address
        """
    ).fetchdf()
    connection.close()
    entries["position_floor_tx_index"] = entries["deploy_tx_index"] + position_lag
    entries["actual_position_delta"] = entries["entry_tx_index"] - entries["deploy_tx_index"]
    return entries


def validate_position_entries(frame: pd.DataFrame, position_lag: int) -> None:
    if frame.empty or frame["token_address"].duplicated().any():
        raise ValueError("Position entries must be nonempty and unique by token")
    covered = frame.loc[frame["covered"]]
    if covered.empty:
        raise ValueError("Position lag produced no same-slot fills")
    if (covered["entry_block_slot"] != covered["deploy_block_slot"]).any():
        raise ValueError("Position-aware entry escaped the deployment slot")
    if (covered["actual_position_delta"] < position_lag).any():
        raise ValueError("Position-aware entry occurs before its transaction-index floor")
    if (covered[["entry_price_usd", "entry_price_sol"]] <= 0).any(axis=None):
        raise ValueError("Position-aware entry contains a nonpositive price")
    missing = frame.loc[~frame["covered"]]
    entry_columns = [
        "entry_slot_index_id",
        "entry_block_slot",
        "entry_tx_index",
        "entry_event_index",
        "entry_block_time",
        "entry_price_usd",
        "entry_price_sol",
        "actual_position_delta",
    ]
    if missing[entry_columns].notna().any(axis=None):
        raise ValueError("Missing position entries contain execution fields")


def write_position_svg(comparison: list[dict[str, object]], path: Path) -> None:
    scenarios = ["gross", "training_median_fee", "training_p90_fee"]
    labels = {
        "gross": "Gross",
        "training_median_fee": "Median fees",
        "training_p90_fee": "P90 fees",
    }
    prior = {str(row["fee_scenario"]): float(row["prior_net_mean_return"]) for row in comparison}
    current = {
        str(row["fee_scenario"]): float(row["position_lag_net_mean_return"]) for row in comparison
    }
    scale = max(max(abs(value) for value in [*prior.values(), *current.values()]), 0.01)
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="430" viewBox="0 0 860 430">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="38" font-family="Arial" font-size="22" font-weight="700" '
        'fill="#0f172a">Zero-slot return after transaction-position lag</text>',
        '<text x="32" y="62" font-family="Arial" font-size="13" fill="#475569">'
        "First post-deploy trade versus training-wallet median position floor</text>",
        '<text x="360" y="96" text-anchor="middle" font-family="Arial" font-size="14" '
        'font-weight="700" fill="#334155">First post-deploy trade</text>',
        '<text x="625" y="96" text-anchor="middle" font-family="Arial" font-size="14" '
        'font-weight="700" fill="#334155">Position lag 112</text>',
    ]
    for index, scenario in enumerate(scenarios):
        y = 126 + index * 88
        elements.append(
            f'<text x="32" y="{y + 34}" font-family="Arial" font-size="14" '
            f'fill="#334155">{labels[scenario]}</text>'
        )
        for x, value in ((255, prior[scenario]), (520, current[scenario])):
            color = "#16a34a" if value >= 0 else "#dc2626"
            opacity = 0.2 + 0.6 * min(abs(value) / scale, 1.0)
            elements.extend(
                [
                    f'<rect x="{x}" y="{y}" width="210" height="58" rx="7" '
                    f'fill="{color}" fill-opacity="{opacity:.3f}"/>',
                    f'<text x="{x + 105}" y="{y + 37}" text-anchor="middle" '
                    f'font-family="Arial" font-size="17" font-weight="700" fill="#0f172a">'
                    f"{100 * value:+.2f}%</text>",
                ]
            )
    elements.extend(
        [
            '<text x="32" y="414" font-family="Arial" font-size="12" fill="#64748b">'
            "Population-weighted validation means; final chronological holdout remains "
            "sealed.</text>",
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def run_position_backtest() -> dict[str, object]:
    for path in (
        CLASSIFICATION_PATH,
        ENTRY_LATENCY_PATH,
        ENTRY_PRICES_PATH,
        WALLET_PATH,
        TRADES_PATH,
        PRIOR_BACKTEST_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    frame = pd.read_parquet(CLASSIFICATION_PATH)
    positive_times = frame.loc[frame["label"] == 1, "decision_time"]
    frame = frame.loc[frame["decision_time"].between(positive_times.min(), positive_times.max())]
    features = [
        column
        for column in frame.select_dtypes(include=["number", "bool"]).columns
        if column not in NON_FEATURE_COLUMNS
    ]
    assert_feature_names_are_pre_decision(features)
    split = chronological_train_validation_test_split(
        frame, time_column="decision_time", validation_fraction=0.2, test_fraction=0.2
    )
    train_end = split.train["decision_time"].max()
    holdout_start = split.test["decision_time"].min()
    outcome_cutoff_epoch = int(holdout_start.timestamp())

    model = build_boosting_model()
    model.fit(split.train[features], split.train["label"])
    probabilities = model.predict_proba(split.validation[features])[:, 1]
    weights = _population_weights(split.validation["label"])
    operating_point = _best_f1_threshold(split.validation["label"], probabilities, weights)
    classifier_metrics = _fixed_threshold_metrics(
        split.validation["label"], probabilities, operating_point["threshold"], weights
    )
    validation = split.validation[
        ["token_address", "decision_time", "label", "blockTime", "blockSlot", "transaction_index"]
    ].copy()
    validation["probability"] = probabilities
    validation["population_weight"] = weights
    validation = validation.rename(
        columns={
            "blockTime": "deploy_block_time",
            "blockSlot": "deploy_block_slot",
            "transaction_index": "deploy_tx_index",
        }
    )
    development = validation.loc[
        (validation["decision_time"] >= JUNE_START) & (validation["decision_time"] < holdout_start)
    ].copy()
    selected = development.loc[development["probability"] >= operating_point["threshold"]].copy()
    if selected.empty or selected["decision_time"].max() >= holdout_start:
        raise ValueError("Selected candidates violate the final holdout boundary")

    position = training_position_parameters(pd.read_parquet(ENTRY_LATENCY_PATH), train_end)
    position_lag = int(position["median"])
    prior_entries = pd.read_parquet(ENTRY_PRICES_PATH)
    prior_entries = selected[["token_address"]].merge(
        prior_entries.loc[prior_entries["requested_delay_slots"] == 0],
        on="token_address",
        how="left",
        validate="one_to_one",
    )
    prior_same_slot = prior_entries.loc[prior_entries["actual_delay_slots"] == 0].copy()
    prior_position_gap = prior_same_slot["entry_tx_index"] - prior_same_slot["deploy_tx_index"]
    entries = build_position_entries(
        selected,
        TRADES_PATH,
        position_lag=position_lag,
        outcome_cutoff_epoch=outcome_cutoff_epoch,
    )
    validate_position_entries(entries, position_lag)
    eligible = entries.loc[
        entries["covered"] & (entries["entry_block_time"] + HOLD_SECONDS < outcome_cutoff_epoch)
    ].copy()
    eligible["exit_target_time"] = eligible["entry_block_time"].astype("int64") + HOLD_SECONDS
    marked = build_exit_marks(
        eligible,
        TRADES_PATH,
        hold_seconds=HOLD_SECONDS,
        outcome_cutoff_epoch=outcome_cutoff_epoch,
    )
    marked["gross_return"] = marked["exit_price_usd"] / marked["entry_price_usd"] - 1
    marked["mark_staleness_seconds"] = marked["exit_target_time"] - marked["exit_block_time"]
    validate_backtest_rows(marked, outcome_cutoff_epoch)

    behavior = training_behavior_parameters(pd.read_parquet(WALLET_PATH), train_end)
    fee_scenarios = {
        "gross": 0.0,
        "training_median_fee": float(behavior["fee_bps"]["roundtrip_median"]),
        "training_p90_fee": float(behavior["fee_bps"]["roundtrip_p90"]),
    }
    notional = float(behavior["first_buy_notional_usd"]["median"])
    current_results = []
    for name, fee_bps in fee_scenarios.items():
        current_results.append(
            {
                "fee_scenario": name,
                "fee_bps": fee_bps,
                **calculate_portfolio_metrics(marked, fee_bps=fee_bps, notional_usd=notional),
            }
        )

    prior = json.loads(PRIOR_BACKTEST_PATH.read_text(encoding="utf-8"))
    prior_rows = {
        str(row["fee_scenario"]): row
        for row in prior["backtest_results"]
        if row["requested_delay_slots"] == 0 and row["execution_policy"] == "all_observed_proxy"
    }
    comparison = []
    for current in current_results:
        name = str(current["fee_scenario"])
        comparison.append(
            {
                "fee_scenario": name,
                "prior_net_mean_return": float(prior_rows[name]["net_mean_return"]),
                "prior_net_median_return": float(prior_rows[name]["net_median_return_unweighted"]),
                "position_lag_net_mean_return": float(current["net_mean_return"]),
                "position_lag_net_median_return": float(current["net_median_return_unweighted"]),
                "mean_return_delta": float(
                    current["net_mean_return"] - prior_rows[name]["net_mean_return"]
                ),
            }
        )
    write_position_svg(comparison, FIGURE_PATH)

    covered = entries.loc[entries["covered"]]
    metrics: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment": "zero_slot_training_median_position_lag",
        "single_hypothesis": (
            "zero-slot profitability survives replacing the first post-deployment trade with "
            "the first same-slot trade no earlier than the training-wallet median transaction "
            "position"
        ),
        "decision": "supported_or_rejected_after_validation_run",
        "development_status": "validation_diagnostic_not_independent_final_estimate",
        "final_holdout_evaluated": False,
        "train_end_utc": train_end.isoformat(),
        "validation_end_utc": split.validation["decision_time"].max().isoformat(),
        "final_holdout_start_utc": holdout_start.isoformat(),
        "max_selected_decision_time_utc": selected["decision_time"].max().isoformat(),
        "max_backtest_outcome_epoch": int(marked["exit_block_time"].max()),
        "classification_dataset": project_relative(CLASSIFICATION_PATH),
        "classification_dataset_sha256": sha256_file(CLASSIFICATION_PATH),
        "entry_latency": project_relative(ENTRY_LATENCY_PATH),
        "entry_latency_sha256": sha256_file(ENTRY_LATENCY_PATH),
        "prior_entry_prices": project_relative(ENTRY_PRICES_PATH),
        "prior_entry_prices_sha256": sha256_file(ENTRY_PRICES_PATH),
        "wallet_source": project_relative(WALLET_PATH),
        "wallet_source_sha256": sha256_file(WALLET_PATH),
        "trades_source": project_relative(TRADES_PATH),
        "trades_source_sha256": sha256_file(TRADES_PATH),
        "prior_backtest": project_relative(PRIOR_BACKTEST_PATH),
        "prior_backtest_sha256": sha256_file(PRIOR_BACKTEST_PATH),
        "model_hyperparameters": BOOSTING_PARAMETERS,
        "feature_names": features,
        "classifier_validation_metrics": classifier_metrics,
        "classifier_operating_point": operating_point,
        "development_june_rows": int(len(development)),
        "selected_sample_rows": int(len(selected)),
        "selected_population_weight": float(selected["population_weight"].sum()),
        "training_wallet_position": position,
        "frozen_position_lag_transactions": position_lag,
        "first_postdeploy_exact_same_slot_position_gap": {
            "rows": int(len(prior_position_gap)),
            "minimum": int(prior_position_gap.min()),
            "p10": float(prior_position_gap.quantile(0.1)),
            "median": float(prior_position_gap.median()),
            "p90": float(prior_position_gap.quantile(0.9)),
            "maximum": int(prior_position_gap.max()),
            "share_at_or_after_frozen_lag": float((prior_position_gap >= position_lag).mean()),
        },
        "entry_definition": (
            "first observed trade in the deployment slot with tx_index at least "
            "deploy_tx_index plus the frozen position lag"
        ),
        "attempted_sample_rows": int(len(entries)),
        "attempted_population_weight": float(entries["population_weight"].sum()),
        "covered_sample_rows": int(len(covered)),
        "covered_population_weight": float(covered["population_weight"].sum()),
        "coverage_rate_sampled": float(len(covered) / len(entries)),
        "coverage_rate_population_weighted": float(
            covered["population_weight"].sum() / entries["population_weight"].sum()
        ),
        "actual_position_delta": {
            "minimum": int(covered["actual_position_delta"].min()),
            "median": float(covered["actual_position_delta"].median()),
            "p90": float(covered["actual_position_delta"].quantile(0.9)),
            "maximum": int(covered["actual_position_delta"].max()),
        },
        "backtest_rows": int(len(marked)),
        "mark_staleness_seconds": {
            "median": float(marked["mark_staleness_seconds"].median()),
            "p90": float(marked["mark_staleness_seconds"].quantile(0.9)),
            "maximum": int(marked["mark_staleness_seconds"].max()),
        },
        "behavior_parameters": behavior,
        "backtest_results": current_results,
        "comparison_to_first_postdeploy_trade": comparison,
        "figure": project_relative(FIGURE_PATH),
        "figure_sha256": sha256_file(FIGURE_PATH),
        "code_parent_commit": git_head(),
        "limitations": [
            "Transaction index is a position proxy, not proof a new replica can observe and "
            "react within the same slot.",
            "The six-second exit remains a last-trade mark rather than a demonstrated "
            "executable sell fill.",
            "The operating threshold is selected on validation and the final chronological "
            "holdout remains sealed.",
            "Population weighting estimates omitted negatives but not their exact portfolio "
            "path dependence.",
        ],
    }
    if all(float(row["net_mean_return"]) > 0 for row in current_results):
        metrics["decision"] = "supported_on_validation_but_not_independent"
    else:
        metrics["decision"] = "rejected_position_lag_breaks_fee_robust_profitability"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    metrics["metrics_path"] = project_relative(METRICS_PATH)
    metrics["metrics_sha256"] = sha256_file(METRICS_PATH)
    append_experiment(metrics)
    return metrics


def main() -> None:
    print(json.dumps(run_position_backtest(), indent=2))


if __name__ == "__main__":
    main()
