"""Compare raw and fee-adjusted signer-outflow selectors in the frozen replica backtest."""

import json

import pandas as pd

from solana_sniper.baseline import (
    NON_FEATURE_COLUMNS,
    _best_f1_threshold,
    _fixed_threshold_metrics,
    _population_weights,
)
from solana_sniper.boosting import BOOSTING_PARAMETERS, build_boosting_model
from solana_sniper.deployment_outflow_proxy import (
    PROXY_FEATURE,
    RAW_FEATURE,
    add_deployment_outflow_proxy,
)
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
ENTRY_PRICES_PATH = PROCESSED_DIR / "replica_entry_prices.parquet"
WALLET_PATH = RAW_DIR / "wallet" / "5brv79e_activity.parquet"
TRADES_PATH = RAW_DIR / "june" / "pumpfun_trades.parquet"
RAW_REPORT_PATH = REPORT_DIR / "replica_validation_backtest.json"
METRICS_PATH = REPORT_DIR / "outflow_replica_backtest.json"
REPORT_PATH = REPORT_DIR.parent / "outflow_replica_backtest.md"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "outflow_replica_backtest.svg"


def backtest_acceptance_decision(comparisons: list[dict[str, object]]) -> dict[str, object]:
    """Apply the predeclared economics criterion to median-fee all-observed rows."""
    if {int(row["requested_delay_slots"]) for row in comparisons} != {0, 1, 2}:
        raise ValueError("comparison must contain exactly the 0/1/2-slot rows")
    improved_delays = [
        int(row["requested_delay_slots"])
        for row in comparisons
        if float(row["proxy_minus_raw_net_mean_return"]) > 0
    ]
    by_delay = {int(row["requested_delay_slots"]): row for row in comparisons}
    zero_slot_solvent = not bool(
        by_delay[0]["fee_adjusted_outflow_proxy"]["insolvent_under_capital_model"]
    )
    raw_insolvencies = sum(
        bool(row["raw_signer_delta"]["insolvent_under_capital_model"]) for row in comparisons
    )
    proxy_insolvencies = sum(
        bool(row["fee_adjusted_outflow_proxy"]["insolvent_under_capital_model"])
        for row in comparisons
    )
    supported = (
        len(improved_delays) >= 2 and zero_slot_solvent and proxy_insolvencies <= raw_insolvencies
    )
    return {
        "decision": (
            "supported_on_predeclared_development_backtest_criterion"
            if supported
            else "rejected_on_predeclared_development_backtest_criterion"
        ),
        "required_improved_delay_count": 2,
        "improved_delay_count": len(improved_delays),
        "improved_delays": improved_delays,
        "zero_slot_solvent_required": True,
        "zero_slot_solvent": zero_slot_solvent,
        "raw_insolvent_delay_count": raw_insolvencies,
        "proxy_insolvent_delay_count": proxy_insolvencies,
        "no_additional_insolvency_required": True,
    }


def _fit_selector(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.Series, dict[str, float], dict[str, float]]:
    model = build_boosting_model()
    model.fit(train[feature_names], train["label"])
    probabilities = pd.Series(
        model.predict_proba(validation[feature_names])[:, 1], index=validation.index
    )
    weights = _population_weights(validation["label"])
    operating_point = _best_f1_threshold(validation["label"], probabilities, weights)
    metrics = _fixed_threshold_metrics(
        validation["label"], probabilities, operating_point["threshold"], weights
    )
    return probabilities, operating_point, metrics


def _selection_overlap(
    raw_selected: pd.DataFrame, proxy_selected: pd.DataFrame
) -> dict[str, object]:
    raw_tokens = set(raw_selected["token_address"])
    proxy_tokens = set(proxy_selected["token_address"])
    intersection = raw_tokens & proxy_tokens
    union = raw_tokens | proxy_tokens
    return {
        "raw_selected_sample_rows": len(raw_selected),
        "raw_selected_population_weight": float(raw_selected["population_weight"].sum()),
        "proxy_selected_sample_rows": len(proxy_selected),
        "proxy_selected_population_weight": float(proxy_selected["population_weight"].sum()),
        "intersection_rows": len(intersection),
        "union_rows": len(union),
        "jaccard": float(len(intersection) / len(union)),
        "raw_only_rows": len(raw_tokens - proxy_tokens),
        "proxy_only_rows": len(proxy_tokens - raw_tokens),
    }


def _frozen_backtest(
    selected: pd.DataFrame,
    entry_prices: pd.DataFrame,
    behavior: dict[str, object],
    outcome_cutoff_epoch: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    attempts = selected.merge(entry_prices, on="token_address", how="left", validate="one_to_many")
    expected_attempts = len(selected) * 3
    if len(attempts) != expected_attempts:
        raise ValueError(f"Expected {expected_attempts} token-delay attempts, got {len(attempts)}")
    attempts["outcome_eligible"] = (
        attempts["covered"]
        & attempts["entry_block_time"].notna()
        & (attempts["entry_block_time"] + HOLD_SECONDS < outcome_cutoff_epoch)
    )
    eligible = attempts.loc[attempts["outcome_eligible"]].copy()
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

    fee_scenarios = {
        "gross": 0.0,
        "training_median_fee": behavior["fee_bps"]["roundtrip_median"],
        "training_p90_fee": behavior["fee_bps"]["roundtrip_p90"],
    }
    notional = float(behavior["first_buy_notional_usd"]["median"])
    results: list[dict[str, object]] = []
    for delay in (0, 1, 2):
        delay_attempts = attempts.loc[attempts["requested_delay_slots"] == delay]
        delay_rows = marked.loc[marked["requested_delay_slots"] == delay]
        for policy, policy_rows in (
            ("all_observed_proxy", delay_rows),
            ("exact_target_slot_only", delay_rows.loc[delay_rows["wait_slots_beyond_target"] == 0]),
        ):
            for fee_name, fee_bps in fee_scenarios.items():
                metrics = calculate_portfolio_metrics(
                    policy_rows, fee_bps=float(fee_bps), notional_usd=notional
                )
                results.append(
                    {
                        "requested_delay_slots": delay,
                        "execution_policy": policy,
                        "fee_scenario": fee_name,
                        "fee_bps": float(fee_bps),
                        "attempted_sample_rows": len(delay_attempts),
                        "attempted_population_weight": float(
                            delay_attempts["population_weight"].sum()
                        ),
                        "coverage_rate_sampled": float(len(policy_rows) / len(delay_attempts)),
                        "coverage_rate_population_weighted": float(
                            policy_rows["population_weight"].sum()
                            / delay_attempts["population_weight"].sum()
                        ),
                        "actual_delay_slots": {
                            "median": float(policy_rows["actual_delay_slots"].median()),
                            "p90": float(policy_rows["actual_delay_slots"].quantile(0.9)),
                            "max": int(policy_rows["actual_delay_slots"].max()),
                        },
                        "wait_beyond_target_slots": {
                            "median": float(policy_rows["wait_slots_beyond_target"].median()),
                            "p90": float(policy_rows["wait_slots_beyond_target"].quantile(0.9)),
                            "max": int(policy_rows["wait_slots_beyond_target"].max()),
                        },
                        **metrics,
                    }
                )
    audit = {
        "attempt_rows": len(attempts),
        "covered_and_cutoff_safe_rows": len(eligible),
        "backtest_rows": len(marked),
        "max_backtest_outcome_epoch": int(marked["exit_block_time"].max()),
        "mark_staleness_seconds": {
            "median": float(marked["mark_staleness_seconds"].median()),
            "p90": float(marked["mark_staleness_seconds"].quantile(0.9)),
            "max": int(marked["mark_staleness_seconds"].max()),
        },
    }
    return results, audit


def _result_key(row: dict[str, object]) -> tuple[int, str, str]:
    return (
        int(row["requested_delay_slots"]),
        str(row["execution_policy"]),
        str(row["fee_scenario"]),
    )


def _primary_comparisons(
    raw_results: list[dict[str, object]], proxy_results: list[dict[str, object]]
) -> list[dict[str, object]]:
    raw_by_key = {_result_key(row): row for row in raw_results}
    proxy_by_key = {_result_key(row): row for row in proxy_results}
    if set(raw_by_key) != set(proxy_by_key):
        raise ValueError("raw and proxy result grids differ")
    comparisons = []
    for delay in (0, 1, 2):
        key = (delay, "all_observed_proxy", "training_median_fee")
        raw = raw_by_key[key]
        proxy = proxy_by_key[key]
        comparisons.append(
            {
                "requested_delay_slots": delay,
                "execution_policy": key[1],
                "fee_scenario": key[2],
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
            }
        )
    return comparisons


def run_outflow_replica_comparison() -> dict[str, object]:
    required_paths = (
        CLASSIFICATION_PATH,
        ENTRY_PRICES_PATH,
        WALLET_PATH,
        TRADES_PATH,
        RAW_REPORT_PATH,
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
    validation = split.validation[["token_address", "decision_time", "label"]].copy()
    validation["population_weight"] = weights
    validation["raw_probability"] = raw_probabilities
    validation["proxy_probability"] = proxy_probabilities
    final_holdout_start = split.test["decision_time"].min()
    development = validation.loc[
        (validation["decision_time"] >= JUNE_START)
        & (validation["decision_time"] < final_holdout_start)
    ].copy()
    raw_selected = development.loc[
        development["raw_probability"] >= raw_operating_point["threshold"]
    ].copy()
    proxy_selected = development.loc[
        development["proxy_probability"] >= proxy_operating_point["threshold"]
    ].copy()
    if proxy_selected.empty or proxy_selected["decision_time"].max() >= final_holdout_start:
        raise ValueError("proxy selection violates the sealed holdout boundary")

    frozen_raw = json.loads(RAW_REPORT_PATH.read_text(encoding="utf-8"))
    if frozen_raw["classifier_validation_metrics"] != raw_metrics:
        raise AssertionError("recomputed raw classifier no longer matches the frozen report")
    if frozen_raw["classifier_operating_point"] != raw_operating_point:
        raise AssertionError("recomputed raw operating point no longer matches the frozen report")
    if frozen_raw["selected_sample_rows"] != len(raw_selected):
        raise AssertionError("recomputed raw selection no longer matches the frozen report")

    train_end = split.train["decision_time"].max()
    behavior = training_behavior_parameters(pd.read_parquet(WALLET_PATH), train_end)
    if behavior["hold_seconds"]["median"] != HOLD_SECONDS:
        raise ValueError("training-only median hold no longer matches the frozen rule")
    entry_prices = pd.read_parquet(ENTRY_PRICES_PATH)
    if entry_prices.duplicated(["token_address", "requested_delay_slots"]).any():
        raise ValueError("entry price token-delay keys must be unique")
    outcome_cutoff_epoch = int(final_holdout_start.timestamp())
    proxy_results, backtest_audit = _frozen_backtest(
        proxy_selected, entry_prices, behavior, outcome_cutoff_epoch
    )
    comparisons = _primary_comparisons(frozen_raw["backtest_results"], proxy_results)
    acceptance = backtest_acceptance_decision(comparisons)

    return {
        "experiment": "fee_adjusted_outflow_replica_backtest_comparison",
        "single_hypothesis": (
            "the higher development precision from replacing raw signer balance delta with "
            "fee-adjusted signer nonfee outflow improves frozen 0/1/2-slot replica economics"
        ),
        "development_status": "validation_diagnostic_not_independent_final_estimate",
        "acceptance_criterion": (
            "at training-median fees under all-observed execution, net weighted mean return "
            "must improve at two or more of 0/1/2 slots, zero-slot must remain solvent, and "
            "the number of insolvent delays must not increase"
        ),
        "acceptance": acceptance,
        "classification_dataset": project_relative(CLASSIFICATION_PATH),
        "classification_dataset_sha256": sha256_file(CLASSIFICATION_PATH),
        "entry_prices": project_relative(ENTRY_PRICES_PATH),
        "entry_prices_sha256": sha256_file(ENTRY_PRICES_PATH),
        "wallet_source": project_relative(WALLET_PATH),
        "wallet_source_sha256": sha256_file(WALLET_PATH),
        "trades_source": project_relative(TRADES_PATH),
        "trades_source_sha256": sha256_file(TRADES_PATH),
        "frozen_raw_report": project_relative(RAW_REPORT_PATH),
        "frozen_raw_report_sha256": sha256_file(RAW_REPORT_PATH),
        "model_hyperparameters": BOOSTING_PARAMETERS,
        "raw_feature_names": raw_features,
        "proxy_feature_names": proxy_features,
        "feature_replacement": {"removed": RAW_FEATURE, "added": PROXY_FEATURE},
        "train_end_utc": train_end.isoformat(),
        "validation_start_utc": split.validation["decision_time"].min().isoformat(),
        "validation_end_utc": split.validation["decision_time"].max().isoformat(),
        "final_holdout_start_utc": final_holdout_start.isoformat(),
        "max_selected_decision_time_utc": proxy_selected["decision_time"].max().isoformat(),
        "final_holdout_evaluated": False,
        "final_holdout_status": "sealed_no_predictions_generated",
        "raw_classifier_validation_metrics": raw_metrics,
        "raw_classifier_operating_point": raw_operating_point,
        "proxy_classifier_validation_metrics": proxy_metrics,
        "proxy_classifier_operating_point": proxy_operating_point,
        "selection_overlap": _selection_overlap(raw_selected, proxy_selected),
        "behavior_parameters": behavior,
        "backtest_audit": backtest_audit,
        "primary_median_fee_comparisons": comparisons,
        "proxy_backtest_results": proxy_results,
        "code_parent_commit": git_head(),
        "limitations": [
            "The first observed trade at or after each target slot is an optimistic entry proxy.",
            "The six-second exit is a last-trade mark, not a proven executable sell fill.",
            "The threshold is selected on validation; the final chronological holdout is sealed.",
            "This comparison does not overturn the separate transaction-position-lag rejection.",
            "Population weighting cannot recover path dependence from unsampled negatives.",
        ],
    }


def render_report(metrics: dict[str, object]) -> str:
    rows = []
    for row in metrics["primary_median_fee_comparisons"]:
        raw = row["raw_signer_delta"]
        proxy = row["fee_adjusted_outflow_proxy"]
        rows.append(
            "| {delay} | {raw_mean:+.2%} | {proxy_mean:+.2%} | {delta:+.2%} | "
            "{raw_median:+.2%} | {proxy_median:+.2%} | {raw_hit:.2%} | {proxy_hit:.2%} | "
            "{raw_dd:.2%} | {proxy_dd:.2%} |".format(
                delay=row["requested_delay_slots"],
                raw_mean=raw["net_mean_return"],
                proxy_mean=proxy["net_mean_return"],
                delta=row["proxy_minus_raw_net_mean_return"],
                raw_median=raw["net_median_return_unweighted"],
                proxy_median=proxy["net_median_return_unweighted"],
                raw_hit=raw["net_hit_rate"],
                proxy_hit=proxy["net_hit_rate"],
                raw_dd=raw["max_drawdown_fraction"],
                proxy_dd=proxy["max_drawdown_fraction"],
            )
        )
    acceptance = metrics["acceptance"]
    raw_classifier = metrics["raw_classifier_validation_metrics"]
    proxy_classifier = metrics["proxy_classifier_validation_metrics"]
    overlap = metrics["selection_overlap"]
    decision_text = (
        "The hypothesis passes the predeclared development criterion."
        if acceptance["decision"].startswith("supported")
        else "The hypothesis fails the predeclared development criterion."
    )
    table_header = (
        "| Delay | Raw mean | Proxy mean | Delta | Raw median | Proxy median | "
        "Raw hit | Proxy hit | Raw MDD | Proxy MDD |"
    )
    return f"""# Fee-adjusted outflow replica comparison

## Decision

{decision_text} Mean return improved for {acceptance["improved_delay_count"]}/3 requested delays;
zero-slot solvency was {acceptance["zero_slot_solvent"]}; insolvent delays changed from
{acceptance["raw_insolvent_delay_count"]} to {acceptance["proxy_insolvent_delay_count"]}.
This is a reproduced validation diagnostic, not an independent final estimate.

{table_header}
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

![Raw versus fee-adjusted selector backtest](figures/outflow_replica_backtest.svg)

## Classifier and selection

The only model change is replacing `{RAW_FEATURE}` with `{PROXY_FEATURE}`. Population-adjusted
validation PR-AUC changed from {raw_classifier["population_adjusted_pr_auc"]:.5f} to
{proxy_classifier["population_adjusted_pr_auc"]:.5f}; precision changed from
{raw_classifier["precision"]:.4f} to {proxy_classifier["precision"]:.4f}, recall from
{raw_classifier["recall"]:.4f} to {proxy_classifier["recall"]:.4f}, and F1 from
{raw_classifier["f1"]:.4f} to {proxy_classifier["f1"]:.4f}. The June development selections
contain {overlap["raw_selected_sample_rows"]} raw and {overlap["proxy_selected_sample_rows"]}
proxy rows with Jaccard overlap {overlap["jaccard"]:.4f}.

## Frozen execution and boundary

- Exit: training-only target-wallet median hold, fixed at six seconds.
- Fees: gross, training median, and training p90 roundtrip fees; all scenarios are in the JSON.
- Entries: first observed trade at or after deploy slot + 0/1/2, excluding deployment transaction.
- Maximum proxy-selected decision time: `{metrics["max_selected_decision_time_utc"]}`.
- Final holdout starts: `{metrics["final_holdout_start_utc"]}` and remains sealed.
- Reproduction: two full deterministic runs matched the complete metrics dictionary exactly.

## Interpretation limits

The entry rule is optimistic and the exit is a mark rather than a guaranteed fill. This result does
not supersede the separate transaction-position-lag test, which rejected zero-slot feasibility.
Post-deployment trades are used only for backtest outcomes, never as classifier features.
"""


def render_svg(metrics: dict[str, object]) -> str:
    comparisons = metrics["primary_median_fee_comparisons"]
    values = [
        float(row[model]["net_mean_return"])
        for row in comparisons
        for model in ("raw_signer_delta", "fee_adjusted_outflow_proxy")
    ]
    minimum = min(values + [0.0])
    maximum = max(values + [0.0])
    span = maximum - minimum or 1.0
    top, height = 78, 220
    zero_y = top + maximum / span * height
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="370" viewBox="0 0 860 370">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Arial" font-size="22" font-weight="700" '
        'fill="#111827">Fee-adjusted selector: replica economics</text>',
        '<text x="28" y="57" font-family="Arial" font-size="13" fill="#4b5563">'
        "Training-median fees; weighted mean return; final holdout sealed</text>",
        f'<line x1="70" y1="{zero_y:.1f}" x2="810" y2="{zero_y:.1f}" stroke="#9ca3af"/>',
    ]
    for index, row in enumerate(comparisons):
        center = 190 + index * 230
        for offset, key, color in (
            (-35, "raw_signer_delta", "#64748b"),
            (10, "fee_adjusted_outflow_proxy", "#0f766e"),
        ):
            value = float(row[key]["net_mean_return"])
            bar_height = abs(value) / span * height
            y = zero_y - bar_height if value >= 0 else zero_y
            elements.extend(
                [
                    f'<rect x="{center + offset}" y="{y:.1f}" width="34" '
                    f'height="{bar_height:.1f}" fill="{color}"/>',
                    f'<text x="{center + offset + 17}" '
                    f'y="{y - 7 if value >= 0 else y + bar_height + 18:.1f}" '
                    f'text-anchor="middle" font-family="Arial" font-size="11">{value:+.1%}</text>',
                ]
            )
        elements.append(
            f'<text x="{center}" y="335" text-anchor="middle" font-family="Arial" '
            f'font-size="13">delay {row["requested_delay_slots"]}</text>'
        )
    elements.extend(
        [
            '<rect x="590" y="26" width="12" height="12" fill="#64748b"/>',
            '<text x="608" y="37" font-family="Arial" font-size="12">Raw delta</text>',
            '<rect x="690" y="26" width="12" height="12" fill="#0f766e"/>',
            '<text x="708" y="37" font-family="Arial" font-size="12">Nonfee outflow</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def main() -> None:
    metrics = run_outflow_replica_comparison()
    reproduced = run_outflow_replica_comparison()
    if reproduced != metrics:
        raise AssertionError("deterministic outflow-replica rerun did not match")
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
