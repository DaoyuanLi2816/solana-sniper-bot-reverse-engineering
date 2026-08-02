"""Test the fee-adjusted selector under the frozen same-slot transaction-position lag."""

import json

import pandas as pd

from solana_sniper.baseline import NON_FEATURE_COLUMNS, _population_weights
from solana_sniper.boosting import BOOSTING_PARAMETERS
from solana_sniper.deployment_outflow_proxy import (
    PROXY_FEATURE,
    RAW_FEATURE,
    add_deployment_outflow_proxy,
)
from solana_sniper.guardrails import assert_feature_names_are_pre_decision
from solana_sniper.manifest import append_experiment, git_head, sha256_file
from solana_sniper.outflow_replica_backtest import _fit_selector, _selection_overlap
from solana_sniper.paths import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, REPORT_DIR, project_relative
from solana_sniper.position_backtest import (
    build_position_entries,
    training_position_parameters,
    validate_position_entries,
)
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
WALLET_PATH = RAW_DIR / "wallet" / "5brv79e_activity.parquet"
TRADES_PATH = RAW_DIR / "june" / "pumpfun_trades.parquet"
RAW_POSITION_REPORT_PATH = REPORT_DIR / "position_lag_validation_backtest.json"
METRICS_PATH = REPORT_DIR / "outflow_position_backtest.json"
REPORT_PATH = REPORT_DIR.parent / "outflow_position_backtest.md"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "outflow_position_backtest.svg"


def position_acceptance_decision(comparisons: list[dict[str, object]]) -> dict[str, object]:
    """Apply the predeclared fee-robust, capital-aware acceptance criterion."""
    by_fee = {str(row["fee_scenario"]): row for row in comparisons}
    expected = {"gross", "training_median_fee", "training_p90_fee"}
    if set(by_fee) != expected:
        raise ValueError("comparison must contain exactly the three frozen fee scenarios")
    median_row = by_fee["training_median_fee"]
    p90_row = by_fee["training_p90_fee"]
    median_proxy = median_row["fee_adjusted_outflow_proxy"]
    p90_proxy = p90_row["fee_adjusted_outflow_proxy"]
    checks = {
        "median_fee_positive_net_mean": float(median_proxy["net_mean_return"]) > 0,
        "p90_fee_positive_net_mean": float(p90_proxy["net_mean_return"]) > 0,
        "median_fee_solvent": not bool(median_proxy["insolvent_under_capital_model"]),
        "p90_fee_solvent": not bool(p90_proxy["insolvent_under_capital_model"]),
        "median_fee_mean_improves_raw": (float(median_row["proxy_minus_raw_net_mean_return"]) > 0),
        "median_fee_drawdown_improves_raw": (
            float(median_row["proxy_minus_raw_max_drawdown_fraction"]) < 0
        ),
    }
    supported = all(checks.values())
    return {
        "decision": (
            "supported_fee_robust_position_lag_feasibility_on_development"
            if supported
            else "rejected_fee_robust_position_lag_feasibility_on_development"
        ),
        "all_checks_passed": supported,
        "checks": checks,
    }


def _comparison_rows(
    raw_results: list[dict[str, object]], proxy_results: list[dict[str, object]]
) -> list[dict[str, object]]:
    raw_by_fee = {str(row["fee_scenario"]): row for row in raw_results}
    proxy_by_fee = {str(row["fee_scenario"]): row for row in proxy_results}
    if set(raw_by_fee) != set(proxy_by_fee):
        raise ValueError("raw and proxy fee grids differ")
    rows = []
    for fee_name in ("gross", "training_median_fee", "training_p90_fee"):
        raw = raw_by_fee[fee_name]
        proxy = proxy_by_fee[fee_name]
        rows.append(
            {
                "fee_scenario": fee_name,
                "fee_bps": float(proxy["fee_bps"]),
                "raw_signer_delta": raw,
                "fee_adjusted_outflow_proxy": proxy,
                "proxy_minus_raw_net_mean_return": float(
                    proxy["net_mean_return"] - raw["net_mean_return"]
                ),
                "proxy_minus_raw_net_median_return": float(
                    proxy["net_median_return_unweighted"] - raw["net_median_return_unweighted"]
                ),
                "proxy_minus_raw_hit_rate": float(proxy["net_hit_rate"] - raw["net_hit_rate"]),
                "proxy_minus_raw_max_drawdown_fraction": float(
                    proxy["max_drawdown_fraction"] - raw["max_drawdown_fraction"]
                ),
                "proxy_minus_raw_total_weighted_pnl_usd": float(
                    proxy["total_weighted_pnl_usd"] - raw["total_weighted_pnl_usd"]
                ),
            }
        )
    return rows


def run_outflow_position_comparison() -> dict[str, object]:
    required_paths = (
        CLASSIFICATION_PATH,
        ENTRY_LATENCY_PATH,
        WALLET_PATH,
        TRADES_PATH,
        RAW_POSITION_REPORT_PATH,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    frame = pd.read_parquet(CLASSIFICATION_PATH)
    if frame["token_address"].duplicated().any():
        raise ValueError("classification token_address must be unique")
    positive_times = frame.loc[frame["label"] == 1, "decision_time"]
    active = frame.loc[frame["decision_time"].between(positive_times.min(), positive_times.max())]
    active = add_deployment_outflow_proxy(active)
    split = chronological_train_validation_test_split(
        active, time_column="decision_time", validation_fraction=0.2, test_fraction=0.2
    )
    numeric_features = [
        column
        for column in active.select_dtypes(include=["number", "bool"]).columns
        if column not in NON_FEATURE_COLUMNS
    ]
    raw_features = [column for column in numeric_features if column != PROXY_FEATURE]
    proxy_features = [PROXY_FEATURE if column == RAW_FEATURE else column for column in raw_features]
    if RAW_FEATURE not in raw_features or len(raw_features) != len(proxy_features):
        raise ValueError("single-feature replacement invariant failed")
    assert_feature_names_are_pre_decision(raw_features)
    assert_feature_names_are_pre_decision(proxy_features)

    raw_probabilities, raw_operating_point, raw_metrics = _fit_selector(
        split.train, split.validation, raw_features
    )
    proxy_probabilities, proxy_operating_point, proxy_metrics = _fit_selector(
        split.train, split.validation, proxy_features
    )
    weights = _population_weights(split.validation["label"])
    validation = split.validation[
        ["token_address", "decision_time", "label", "blockTime", "blockSlot", "transaction_index"]
    ].copy()
    validation["population_weight"] = weights
    validation["raw_probability"] = raw_probabilities
    validation["proxy_probability"] = proxy_probabilities
    validation = validation.rename(
        columns={
            "blockTime": "deploy_block_time",
            "blockSlot": "deploy_block_slot",
            "transaction_index": "deploy_tx_index",
        }
    )
    holdout_start = split.test["decision_time"].min()
    development = validation.loc[
        (validation["decision_time"] >= JUNE_START) & (validation["decision_time"] < holdout_start)
    ].copy()
    raw_selected = development.loc[
        development["raw_probability"] >= raw_operating_point["threshold"]
    ].copy()
    proxy_selected = development.loc[
        development["proxy_probability"] >= proxy_operating_point["threshold"]
    ].copy()
    if proxy_selected.empty or proxy_selected["decision_time"].max() >= holdout_start:
        raise ValueError("proxy selection violates the sealed holdout boundary")

    frozen_raw = json.loads(RAW_POSITION_REPORT_PATH.read_text(encoding="utf-8"))
    if frozen_raw["classifier_validation_metrics"] != raw_metrics:
        raise AssertionError("recomputed raw classifier no longer matches frozen position report")
    if frozen_raw["classifier_operating_point"] != raw_operating_point:
        raise AssertionError("recomputed raw operating point no longer matches frozen report")
    if frozen_raw["selected_sample_rows"] != len(raw_selected):
        raise AssertionError("recomputed raw selection no longer matches frozen report")

    train_end = split.train["decision_time"].max()
    position = training_position_parameters(pd.read_parquet(ENTRY_LATENCY_PATH), train_end)
    position_lag = int(position["median"])
    if position_lag != int(frozen_raw["frozen_position_lag_transactions"]):
        raise AssertionError("training-derived position lag changed from frozen raw report")
    outcome_cutoff_epoch = int(holdout_start.timestamp())
    entries = build_position_entries(
        proxy_selected,
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
    proxy_results = [
        {
            "fee_scenario": name,
            "fee_bps": fee_bps,
            **calculate_portfolio_metrics(marked, fee_bps=fee_bps, notional_usd=notional),
        }
        for name, fee_bps in fee_scenarios.items()
    ]
    comparisons = _comparison_rows(frozen_raw["backtest_results"], proxy_results)
    acceptance = position_acceptance_decision(comparisons)
    covered = entries.loc[entries["covered"]]

    return {
        "experiment": "fee_adjusted_outflow_position_lag_comparison",
        "single_hypothesis": (
            "the higher development precision from fee-adjusted signer nonfee outflow restores "
            "fee-robust zero-slot replica feasibility at the frozen training-wallet position lag"
        ),
        "development_status": "validation_diagnostic_not_independent_final_estimate",
        "acceptance_criterion": (
            "at the frozen 112-transaction position lag, training-median and p90 fee net mean "
            "returns must both be positive and solvent; median-fee mean and drawdown must both "
            "improve versus the raw selector"
        ),
        "acceptance": acceptance,
        "final_holdout_evaluated": False,
        "final_holdout_status": "sealed_no_predictions_generated",
        "train_end_utc": train_end.isoformat(),
        "validation_start_utc": split.validation["decision_time"].min().isoformat(),
        "validation_end_utc": split.validation["decision_time"].max().isoformat(),
        "final_holdout_start_utc": holdout_start.isoformat(),
        "max_selected_decision_time_utc": proxy_selected["decision_time"].max().isoformat(),
        "classification_dataset": project_relative(CLASSIFICATION_PATH),
        "classification_dataset_sha256": sha256_file(CLASSIFICATION_PATH),
        "entry_latency": project_relative(ENTRY_LATENCY_PATH),
        "entry_latency_sha256": sha256_file(ENTRY_LATENCY_PATH),
        "wallet_source": project_relative(WALLET_PATH),
        "wallet_source_sha256": sha256_file(WALLET_PATH),
        "trades_source": project_relative(TRADES_PATH),
        "trades_source_sha256": sha256_file(TRADES_PATH),
        "frozen_raw_position_report": project_relative(RAW_POSITION_REPORT_PATH),
        "frozen_raw_position_report_sha256": sha256_file(RAW_POSITION_REPORT_PATH),
        "model_hyperparameters": BOOSTING_PARAMETERS,
        "raw_feature_names": raw_features,
        "proxy_feature_names": proxy_features,
        "feature_replacement": {"removed": RAW_FEATURE, "added": PROXY_FEATURE},
        "raw_classifier_validation_metrics": raw_metrics,
        "raw_classifier_operating_point": raw_operating_point,
        "proxy_classifier_validation_metrics": proxy_metrics,
        "proxy_classifier_operating_point": proxy_operating_point,
        "selection_overlap": _selection_overlap(raw_selected, proxy_selected),
        "training_wallet_position": position,
        "frozen_position_lag_transactions": position_lag,
        "entry_definition": (
            "first observed trade in the deployment slot with tx_index at least "
            "deploy_tx_index plus the frozen position lag"
        ),
        "attempted_sample_rows": len(entries),
        "attempted_population_weight": float(entries["population_weight"].sum()),
        "covered_sample_rows": len(covered),
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
        "backtest_rows": len(marked),
        "max_backtest_outcome_epoch": int(marked["exit_block_time"].max()),
        "mark_staleness_seconds": {
            "median": float(marked["mark_staleness_seconds"].median()),
            "p90": float(marked["mark_staleness_seconds"].quantile(0.9)),
            "maximum": int(marked["mark_staleness_seconds"].max()),
        },
        "behavior_parameters": behavior,
        "proxy_backtest_results": proxy_results,
        "comparison_to_raw_position_lag": comparisons,
        "code_parent_commit": git_head(),
        "limitations": [
            "Transaction index is a position proxy, not proof of within-slot observability.",
            "The six-second exit is a last-trade mark, not a demonstrated executable sell fill.",
            "The threshold is selected on validation and the final holdout remains sealed.",
            "Selection-dependent coverage differs between raw and proxy candidate sets.",
            "Population weighting cannot recover exact path dependence of unsampled negatives.",
        ],
    }


def render_report(metrics: dict[str, object]) -> str:
    table_rows = []
    for row in metrics["comparison_to_raw_position_lag"]:
        raw = row["raw_signer_delta"]
        proxy = row["fee_adjusted_outflow_proxy"]
        table_rows.append(
            "| {fee} | {raw_mean:+.2%} | {proxy_mean:+.2%} | {raw_median:+.2%} | "
            "{proxy_median:+.2%} | {raw_hit:.2%} | {proxy_hit:.2%} | {raw_dd:.2%} | "
            "{proxy_dd:.2%} | {raw_solvent} | {proxy_solvent} |".format(
                fee=row["fee_scenario"],
                raw_mean=raw["net_mean_return"],
                proxy_mean=proxy["net_mean_return"],
                raw_median=raw["net_median_return_unweighted"],
                proxy_median=proxy["net_median_return_unweighted"],
                raw_hit=raw["net_hit_rate"],
                proxy_hit=proxy["net_hit_rate"],
                raw_dd=raw["max_drawdown_fraction"],
                proxy_dd=proxy["max_drawdown_fraction"],
                raw_solvent=not raw["insolvent_under_capital_model"],
                proxy_solvent=not proxy["insolvent_under_capital_model"],
            )
        )
    decision = metrics["acceptance"]
    overlap = metrics["selection_overlap"]
    raw_classifier = metrics["raw_classifier_validation_metrics"]
    proxy_classifier = metrics["proxy_classifier_validation_metrics"]
    verdict = "passes" if decision["all_checks_passed"] else "fails"
    header = (
        "| Fee | Raw mean | Proxy mean | Raw median | Proxy median | Raw hit | Proxy hit | "
        "Raw MDD | Proxy MDD | Raw solvent | Proxy solvent |"
    )
    return f"""# Fee-adjusted selector under transaction-position lag

## Decision

The hypothesis **{verdict}** the predeclared development criterion. The frozen entry floor is
{metrics["frozen_position_lag_transactions"]} transactions after deployment. This is a reproduced
validation diagnostic; the final chronological holdout remains sealed.

{header}
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
{chr(10).join(table_rows)}

![Position-lag raw versus fee-adjusted selector](figures/outflow_position_backtest.svg)

## Classifier and candidate coverage

The only feature change is `{RAW_FEATURE}` to `{PROXY_FEATURE}`. Population-adjusted validation
PR-AUC is {raw_classifier["population_adjusted_pr_auc"]:.5f} versus
{proxy_classifier["population_adjusted_pr_auc"]:.5f}; precision is
{raw_classifier["precision"]:.4f} versus {proxy_classifier["precision"]:.4f}; recall is
{raw_classifier["recall"]:.4f} versus {proxy_classifier["recall"]:.4f}; and F1 is
{raw_classifier["f1"]:.4f} versus {proxy_classifier["f1"]:.4f}.

The raw selector chooses {overlap["raw_selected_sample_rows"]} June development rows and the proxy
chooses {overlap["proxy_selected_sample_rows"]}, with Jaccard overlap {overlap["jaccard"]:.4f}.
The proxy has {metrics["covered_sample_rows"]}/{metrics["attempted_sample_rows"]} sampled same-slot
fills ({metrics["coverage_rate_sampled"]:.2%}); population-weighted coverage is
{metrics["coverage_rate_population_weighted"]:.2%}.

## Boundary and interpretation

- Position and wallet behavior parameters use only events through `{metrics["train_end_utc"]}`.
- Maximum selected decision time is `{metrics["max_selected_decision_time_utc"]}`.
- Final holdout starts at `{metrics["final_holdout_start_utc"]}` and has no predictions.
- Two complete deterministic runs match the full metrics dictionary exactly.
- Transaction position and six-second exit remain proxies, not guaranteed executable fills.
"""


def render_svg(metrics: dict[str, object]) -> str:
    comparisons = metrics["comparison_to_raw_position_lag"]
    values = [
        float(row[model]["net_mean_return"])
        for row in comparisons
        for model in ("raw_signer_delta", "fee_adjusted_outflow_proxy")
    ]
    minimum = min(values + [0.0])
    maximum = max(values + [0.0])
    span = maximum - minimum or 1.0
    top, height = 86, 220
    zero_y = top + maximum / span * height
    labels = {
        "gross": "Gross",
        "training_median_fee": "Median fees",
        "training_p90_fee": "P90 fees",
    }
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="880" height="390" viewBox="0 0 880 390">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Arial" font-size="22" font-weight="700" '
        'fill="#111827">Fee-adjusted selector at position lag 112</text>',
        '<text x="28" y="58" font-family="Arial" font-size="13" fill="#4b5563">'
        "Weighted net mean return; validation only; final holdout sealed</text>",
        f'<line x1="70" y1="{zero_y:.1f}" x2="830" y2="{zero_y:.1f}" stroke="#9ca3af"/>',
    ]
    for index, row in enumerate(comparisons):
        center = 190 + index * 240
        for offset, key, color in (
            (-38, "raw_signer_delta", "#64748b"),
            (8, "fee_adjusted_outflow_proxy", "#0f766e"),
        ):
            value = float(row[key]["net_mean_return"])
            bar_height = abs(value) / span * height
            y = zero_y - bar_height if value >= 0 else zero_y
            label_y = y - 7 if value >= 0 else y + bar_height + 18
            elements.extend(
                [
                    f'<rect x="{center + offset}" y="{y:.1f}" width="36" '
                    f'height="{bar_height:.1f}" fill="{color}"/>',
                    f'<text x="{center + offset + 18}" y="{label_y:.1f}" '
                    f'text-anchor="middle" font-family="Arial" font-size="11">{value:+.1%}</text>',
                ]
            )
        elements.append(
            f'<text x="{center}" y="350" text-anchor="middle" font-family="Arial" '
            f'font-size="13">{labels[row["fee_scenario"]]}</text>'
        )
    elements.extend(
        [
            '<rect x="615" y="27" width="12" height="12" fill="#64748b"/>',
            '<text x="633" y="38" font-family="Arial" font-size="12">Raw delta</text>',
            '<rect x="715" y="27" width="12" height="12" fill="#0f766e"/>',
            '<text x="733" y="38" font-family="Arial" font-size="12">Nonfee outflow</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def main() -> None:
    metrics = run_outflow_position_comparison()
    reproduced = run_outflow_position_comparison()
    if reproduced != metrics:
        raise AssertionError("deterministic outflow-position rerun did not match")
    metrics["reproduction_verification"] = {
        "verified": True,
        "run_count": 2,
        "comparison": "complete_metrics_dictionary_exact_match",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_report(metrics), encoding="utf-8")
    FIGURE_PATH.write_text(render_svg(metrics), encoding="utf-8")
    append_experiment(
        {
            **metrics,
            "metrics_path": project_relative(METRICS_PATH),
            "metrics_sha256": sha256_file(METRICS_PATH),
            "report_path": project_relative(REPORT_PATH),
            "report_sha256": sha256_file(REPORT_PATH),
            "figure_path": project_relative(FIGURE_PATH),
            "figure_sha256": sha256_file(FIGURE_PATH),
        }
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
