"""Audit zero-slot replica feasibility with a training-derived transaction-position lag."""

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
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
CLASSIFICATION_MANIFEST_PATH = PROCESSED_DIR / "creator_history_manifest.json"
ENTRY_LATENCY_PATH = PROCESSED_DIR / "entry_latency.parquet"
ENTRY_PRICES_PATH = PROCESSED_DIR / "replica_entry_prices.parquet"
ENTRY_PRICES_METRICS_PATH = REPORT_DIR / "entry_prices_metrics.json"
WALLET_PATH = RAW_DIR / "wallet" / "5brv79e_activity.parquet"
TRADES_PATH = RAW_DIR / "june" / "pumpfun_trades.parquet"
PRIOR_BACKTEST_PATH = REPORT_DIR / "replica_validation_backtest.json"
METRICS_PATH = REPORT_DIR / "position_lag_validation_backtest.json"
REPORT_PATH = REPORT_DIR.parent / "position_lag_validation_backtest.md"
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


def position_acceptance_decision(results: list[dict[str, object]]) -> dict[str, object]:
    """Apply the predeclared median-fee, capital-aware feasibility criterion."""
    rows = [row for row in results if row["fee_scenario"] == "training_median_fee"]
    if len(rows) != 1:
        raise ValueError("position results must contain one training-median-fee row")
    row = rows[0]
    checks = {
        "positive_population_weighted_mean": float(row["net_mean_return"]) > 0,
        "positive_unweighted_median": float(row["net_median_return_unweighted"]) > 0,
        "capital_path_solvent": not bool(row["insolvent_under_capital_model"]),
    }
    passed = all(checks.values())
    return {
        "decision": (
            "supported_position_lag_remains_fee_robust_and_solvent"
            if passed
            else "rejected_position_lag_fails_fee_robust_capital_criterion"
        ),
        "criterion": (
            "under training-median roundtrip fees, the population-weighted mean and "
            "unweighted median must both be strictly positive and the fixed-notional "
            "capital path must remain solvent"
        ),
        "checks": checks,
        "all_checks_passed": passed,
    }


def render_position_svg(comparison: list[dict[str, object]], position_lag: int) -> str:
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
        f'font-weight="700" fill="#334155">Position lag {position_lag}</text>',
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
    return "\n".join(elements) + "\n"


def run_position_backtest() -> dict[str, object]:
    required_paths = (
        CLASSIFICATION_PATH,
        CLASSIFICATION_MANIFEST_PATH,
        ENTRY_LATENCY_PATH,
        ENTRY_PRICES_PATH,
        ENTRY_PRICES_METRICS_PATH,
        WALLET_PATH,
        TRADES_PATH,
        PRIOR_BACKTEST_PATH,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    frame = pd.read_parquet(CLASSIFICATION_PATH)
    required_columns = {
        "label",
        "decision_time",
        "token_address",
        "tx_hash",
        "blockTime",
        "blockSlot",
        "transaction_index",
        "decision_hour_utc",
        "decision_weekday_utc",
        "creator_prior_deploy_count",
        "creator_prior_active_slot_count",
        "creator_observed_age_seconds",
        "creator_seconds_since_previous_deploy",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"classification dataset is missing columns: {missing_columns}")
    if frame.empty or frame["token_address"].duplicated().any():
        raise ValueError("classification tokens must be nonempty and unique")
    if frame["label"].nunique() != 2:
        raise ValueError("classification dataset must contain both classes")
    classification_sha256 = sha256_file(CLASSIFICATION_PATH)
    history_manifest = json.loads(CLASSIFICATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if history_manifest["sha256"] != classification_sha256:
        raise ValueError("strict-history manifest hash does not match classification dataset")
    if history_manifest["rows"] != len(frame):
        raise ValueError("strict-history manifest row count does not match")
    if history_manifest["unique_tokens"] != frame["token_address"].nunique():
        raise ValueError("strict-history manifest unique-token count does not match")
    if history_manifest["strict_time_violations"] != 0:
        raise ValueError("strict-history manifest reports time-boundary violations")
    canonical_time = pd.to_datetime(frame["decision_time"], utc=True)
    utc_hour_mismatches = int(frame["decision_hour_utc"].ne(canonical_time.dt.hour).sum())
    utc_weekday_mismatches = int(
        frame["decision_weekday_utc"].ne(canonical_time.dt.dayofweek).sum()
    )
    if utc_hour_mismatches or utc_weekday_mismatches:
        raise ValueError("classification dataset contains noncanonical UTC clock features")

    positive_times = frame.loc[frame["label"] == 1, "decision_time"]
    active = frame.loc[
        frame["decision_time"].between(positive_times.min(), positive_times.max())
    ].copy()
    features = [
        column
        for column in active.select_dtypes(include=["number", "bool"]).columns
        if column not in NON_FEATURE_COLUMNS
    ]
    assert_feature_names_are_pre_decision(features)
    split = chronological_train_validation_test_split(
        active, time_column="decision_time", validation_fraction=0.2, test_fraction=0.2
    )
    for name, partition in (
        ("train", split.train),
        ("validation", split.validation),
        ("final_holdout", split.test),
    ):
        if partition["label"].nunique() != 2:
            raise ValueError(f"{name} partition does not contain both classes")
    train_end = split.train["decision_time"].max()
    holdout_start = split.test["decision_time"].min()
    if not split.validation["decision_time"].max() < holdout_start:
        raise AssertionError("validation overlaps the final holdout")
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

    latency = pd.read_parquet(ENTRY_LATENCY_PATH)
    if latency.empty or latency["token_address"].duplicated().any():
        raise ValueError("entry latency must be nonempty and unique by token")
    positive_token_times = frame.loc[frame["label"] == 1].set_index("token_address")[
        "decision_time"
    ]
    latency_token_times = latency.set_index("token_address")["decision_time"]
    latency_time_mismatches = int(
        pd.to_datetime(latency_token_times, utc=True)
        .ne(pd.to_datetime(positive_token_times.loc[latency_token_times.index], utc=True))
        .sum()
    )
    if latency_time_mismatches:
        raise ValueError("entry-latency decision times differ from corrected classification times")
    position = training_position_parameters(latency, train_end)
    position_lag = int(position["median"])
    prior_entries = pd.read_parquet(ENTRY_PRICES_PATH)
    entry_metrics = json.loads(ENTRY_PRICES_METRICS_PATH.read_text(encoding="utf-8"))
    entry_prices_sha256 = sha256_file(ENTRY_PRICES_PATH)
    if entry_metrics["classification_sha256"] != classification_sha256:
        raise ValueError("entry-price audit is tied to a different classification dataset")
    if entry_metrics["output_sha256"] != entry_prices_sha256:
        raise ValueError("entry-price audit hash does not match the entry-price table")
    if entry_metrics["trades_sha256"] != sha256_file(TRADES_PATH):
        raise ValueError("entry-price audit trades hash does not match the source")
    if entry_metrics["rows"] != len(prior_entries):
        raise ValueError("entry-price audit row count does not match")
    entry_token_times = prior_entries.drop_duplicates("token_address").set_index("token_address")[
        "decision_time"
    ]
    classification_token_times = frame.set_index("token_address")["decision_time"]
    entry_time_mismatches = int(
        pd.to_datetime(entry_token_times, utc=True)
        .ne(pd.to_datetime(classification_token_times.loc[entry_token_times.index], utc=True))
        .sum()
    )
    if entry_time_mismatches:
        raise ValueError("entry-price decision times differ from corrected classification times")
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
    if not np.isfinite(marked["gross_return"]).all():
        raise ValueError("position backtest returns are not finite")

    behavior = training_behavior_parameters(pd.read_parquet(WALLET_PATH), train_end)
    if behavior["hold_seconds"]["median"] != HOLD_SECONDS:
        raise ValueError("Training-only median hold does not match the frozen six-second rule")
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
    if prior["classification_dataset_sha256"] != classification_sha256:
        raise ValueError("prior replica report is tied to a different classification dataset")
    if prior["final_holdout_evaluated"]:
        raise ValueError("prior replica report unexpectedly evaluates the final holdout")
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
    acceptance = position_acceptance_decision(current_results)
    figure_svg = render_position_svg(comparison, position_lag)

    covered = entries.loc[entries["covered"]]
    metrics: dict[str, object] = {
        "experiment": "corrected_zero_slot_training_median_position_lag",
        "single_hypothesis": (
            "zero-slot profitability survives replacing the first post-deployment trade with "
            "the first same-slot trade no earlier than the training-wallet median transaction "
            "position"
        ),
        "predeclared_acceptance_criterion": acceptance["criterion"],
        "acceptance": acceptance,
        "development_status": "validation_diagnostic_not_independent_final_estimate",
        "final_holdout_evaluated": False,
        "final_holdout_status": "sealed_no_predictions_generated",
        "train_end_utc": train_end.isoformat(),
        "validation_start_utc": split.validation["decision_time"].min().isoformat(),
        "validation_end_utc": split.validation["decision_time"].max().isoformat(),
        "final_holdout_start_utc": holdout_start.isoformat(),
        "max_predicted_decision_time_utc": development["decision_time"].max().isoformat(),
        "max_selected_decision_time_utc": selected["decision_time"].max().isoformat(),
        "max_backtest_outcome_epoch": int(marked["exit_block_time"].max()),
        "max_backtest_outcome_utc": pd.Timestamp(
            int(marked["exit_block_time"].max()), unit="s", tz="UTC"
        ).isoformat(),
        "classification_dataset": project_relative(CLASSIFICATION_PATH),
        "classification_dataset_sha256": classification_sha256,
        "classification_manifest": project_relative(CLASSIFICATION_MANIFEST_PATH),
        "classification_manifest_sha256": sha256_file(CLASSIFICATION_MANIFEST_PATH),
        "entry_latency": project_relative(ENTRY_LATENCY_PATH),
        "entry_latency_sha256": sha256_file(ENTRY_LATENCY_PATH),
        "prior_entry_prices": project_relative(ENTRY_PRICES_PATH),
        "prior_entry_prices_sha256": entry_prices_sha256,
        "entry_prices_metrics": project_relative(ENTRY_PRICES_METRICS_PATH),
        "entry_prices_metrics_sha256": sha256_file(ENTRY_PRICES_METRICS_PATH),
        "wallet_source": project_relative(WALLET_PATH),
        "wallet_source_sha256": sha256_file(WALLET_PATH),
        "trades_source": project_relative(TRADES_PATH),
        "trades_source_sha256": sha256_file(TRADES_PATH),
        "prior_backtest": project_relative(PRIOR_BACKTEST_PATH),
        "prior_backtest_sha256": sha256_file(PRIOR_BACKTEST_PATH),
        "source_role": "post_deployment_prices_used_only_for_backtest_entries_and_marks",
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
        "figure_sha256": hashlib.sha256(figure_svg.encode("utf-8")).hexdigest(),
        "schema": {
            "classification_rows": int(len(frame)),
            "classification_active_rows": int(len(active)),
            "classification_unique_tokens": int(frame["token_address"].nunique()),
            "classification_positive_rows": int((frame["label"] == 1).sum()),
            "classification_negative_rows": int((frame["label"] == 0).sum()),
            "strict_time_violations": int(history_manifest["strict_time_violations"]),
            "utc_hour_mismatch_rows": utc_hour_mismatches,
            "utc_weekday_mismatch_rows": utc_weekday_mismatches,
            "entry_latency_rows": int(len(latency)),
            "entry_latency_unique_tokens": int(latency["token_address"].nunique()),
            "entry_latency_decision_time_mismatch_rows": latency_time_mismatches,
            "entry_price_rows": int(len(prior_entries)),
            "entry_price_decision_time_mismatch_rows": entry_time_mismatches,
            "position_entry_rows": int(len(entries)),
            "position_entry_unique_tokens": int(entries["token_address"].nunique()),
            "backtest_unique_tokens": int(marked["token_address"].nunique()),
        },
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
    return metrics


def render_report(metrics: dict[str, object]) -> str:
    position = metrics["training_wallet_position"]
    classifier = metrics["classifier_validation_metrics"]
    rows = {row["fee_scenario"]: row for row in metrics["backtest_results"]}
    comparison = {
        row["fee_scenario"]: row for row in metrics["comparison_to_first_postdeploy_trade"]
    }
    acceptance = metrics["acceptance"]
    verdict = "passes" if acceptance["all_checks_passed"] else "fails"
    table_rows = []
    for label, key in (
        ("Gross", "gross"),
        ("Training median", "training_median_fee"),
        ("Training p90", "training_p90_fee"),
    ):
        row = rows[key]
        insolvent = "yes" if row["insolvent_under_capital_model"] else "no"
        table_rows.append(
            f"| {label} | {row['net_mean_return']:+.2%} | "
            f"{row['net_median_return_unweighted']:+.2%} | "
            f"{row['net_hit_rate']:.2%} | {row['max_drawdown_fraction']:.2%} | "
            f"{insolvent} |"
        )
    result_table = "\n".join(table_rows)
    return f"""# Corrected zero-slot transaction-position feasibility (validation only)

## Hypothesis and frozen policy

This corrected run asks whether the raw signer-delta plus strict creator-history selector remains
economically viable when zero-slot entry uses the target wallet's training-derived transaction
position instead of the first observed post-deployment trade. The classifier uses only fields
available at token deployment (`t_decision`). Post-deployment trades are used only for entry and
six-second exit marks. The final chronological holdout remains sealed.

- Corrected classification SHA-256: `{metrics["classification_dataset_sha256"]}`.
- Training cutoff: `{metrics["train_end_utc"]}`; validation ends
  `{metrics["validation_end_utc"]}`; final holdout starts
  `{metrics["final_holdout_start_utc"]}`.
- Classifier: PR-AUC {classifier["population_adjusted_pr_auc"]:.5f}, precision
  {classifier["precision"]:.5f}, recall {classifier["recall"]:.5f}, F1
  {classifier["f1"]:.5f}, threshold {classifier["threshold"]:.6f}.
- Frozen entry floor: deployment transaction index plus
  {metrics["frozen_position_lag_transactions"]} positions, the median of
  {position["rows"]:,} training-period same-slot target-wallet buys (p10 {position["p10"]:.0f},
  p90 {position["p90"]:.0f}).
- Exit: last observed trade mark no later than entry plus six seconds.
- Fees and notional: derived only from target-wallet events through the training cutoff.

## Execution coverage

The policy attempts {metrics["attempted_sample_rows"]:,} selected validation candidates (population
weight {metrics["attempted_population_weight"]:,.0f}) and obtains same-slot fills for
{metrics["covered_sample_rows"]:,} (population weight {metrics["covered_population_weight"]:,.0f}).
Coverage is {metrics["coverage_rate_sampled"]:.2%} by sampled rows and
{metrics["coverage_rate_population_weighted"]:.2%} after population weighting. Actual filled
position gaps have median {metrics["actual_position_delta"]["median"]:.0f} and p90
{metrics["actual_position_delta"]["p90"]:.1f}. This is a transaction-position proxy, not proof of
live same-slot reaction.

## Results

| fee scenario | weighted mean | unweighted median | weighted hit rate | max drawdown | insolvent |
|:---|---:|---:|---:|---:|:---:|
{result_table}

The transaction-position adjustment changes the median-fee weighted mean by
{comparison["training_median_fee"]["mean_return_delta"]:+.2%} versus the corrected first-observed
trade proxy. Under the predeclared criterion, the executable-position candidate **{verdict}**:
both weighted mean and unweighted median must be positive at training-median fees, and the capital
path must remain solvent. Decision: `{acceptance["decision"]}`.

![Corrected position-lag comparison](figures/position_lag_validation_backtest.svg)

## Integrity and limits

The corrected UTC hour and weekday mismatch counts are both zero, strict-history violations are
zero, selected decisions end at `{metrics["max_selected_decision_time_utc"]}`, and the latest
post-deployment outcome mark is `{metrics["max_backtest_outcome_utc"]}`, strictly before the sealed
holdout. Two complete deterministic runs must match before artifacts are written. The six-second
exit is still a mark rather than a demonstrated executable sell; population weighting estimates
omitted negatives but cannot recover their exact portfolio path.
"""


def main() -> None:
    first = run_position_backtest()
    second = run_position_backtest()
    if first != second:
        raise AssertionError("deterministic position backtest rerun did not match")
    metrics = first
    report = render_report(metrics)
    figure = render_position_svg(
        metrics["comparison_to_first_postdeploy_trade"],
        int(metrics["frozen_position_lag_transactions"]),
    )
    if hashlib.sha256(figure.encode("utf-8")).hexdigest() != metrics["figure_sha256"]:
        raise AssertionError("position figure hash changed between validation and write")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")
    FIGURE_PATH.write_text(figure, encoding="utf-8", newline="\n")
    append_experiment(
        {
            **metrics,
            "metrics_path": project_relative(METRICS_PATH),
            "metrics_sha256": sha256_file(METRICS_PATH),
            "report_path": project_relative(REPORT_PATH),
            "report_sha256": sha256_file(REPORT_PATH),
        }
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
