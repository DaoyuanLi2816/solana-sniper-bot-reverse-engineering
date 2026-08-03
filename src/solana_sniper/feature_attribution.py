"""Rank pre-decision features without opening the final chronological holdout."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from solana_sniper.baseline import _population_weights
from solana_sniper.boosting import BOOSTING_PARAMETERS
from solana_sniper.creator_history_stability import (
    CREATOR_HISTORY_FEATURES,
    _numeric_features,
    _validation_metrics,
    expanding_time_folds,
)
from solana_sniper.guardrails import assert_feature_names_are_pre_decision
from solana_sniper.manifest import append_experiment, git_head, sha256_file
from solana_sniper.paths import PROCESSED_DIR, REPORT_DIR, project_relative
from solana_sniper.splits import chronological_train_validation_test_split

ATTRIBUTION_SEED = 20260802
TEMPORAL_REPEATS = 3
STANDARD_REPEATS = 5

FEATURE_GROUPS = {
    "creator_history": CREATOR_HISTORY_FEATURES,
    "fee_and_compute": [
        "tx_fee_lamports",
        "priority_fee_lamports",
        "compute_units",
        "cost_units",
    ],
    "transaction_structure": [
        "transaction_index",
        "outer_instruction_count",
        "inner_instruction_count",
        "account_count",
        "signature_count",
        "log_message_count",
        "pre_token_balance_count",
        "post_token_balance_count",
        "signer_lamport_delta",
    ],
    "metadata": [
        "metadata_name_length",
        "metadata_symbol_length",
        "metadata_uri_length",
        "metadata_uri_is_ipfs",
        "metadata_present",
    ],
    "decision_time": ["decision_hour_utc", "decision_weekday_utc"],
    "transaction_error": ["transaction_error"],
}

FEATURE_LABELS = {
    "creator_prior_deploy_count": "Prior signer deploy count",
    "creator_prior_active_slot_count": "Prior signer active-slot count",
    "creator_observed_age_seconds": "Observed signer deploy age",
    "creator_seconds_since_previous_deploy": "Seconds since prior deploy",
    "tx_fee_lamports": "Transaction fee",
    "priority_fee_lamports": "Priority fee",
    "compute_units": "Compute units",
    "cost_units": "Cost units",
    "transaction_index": "Transaction position",
    "outer_instruction_count": "Outer instruction count",
    "inner_instruction_count": "Inner instruction count",
    "account_count": "Account count",
    "signature_count": "Signature count",
    "log_message_count": "Log message count",
    "pre_token_balance_count": "Pre-token balance count",
    "post_token_balance_count": "Post-token balance count",
    "signer_lamport_delta": "Signer lamport delta",
    "metadata_name_length": "Metadata name length",
    "metadata_symbol_length": "Metadata symbol length",
    "metadata_uri_length": "Metadata URI length",
    "metadata_uri_is_ipfs": "Metadata URI is IPFS",
    "metadata_present": "Metadata present",
    "transaction_error": "Transaction error",
    "decision_hour_utc": "Decision hour UTC",
    "decision_weekday_utc": "Decision weekday UTC",
}


def validate_feature_groups(features: list[str]) -> None:
    """Require a disjoint and complete grouping of the model features."""
    grouped = [feature for columns in FEATURE_GROUPS.values() for feature in columns]
    duplicates = sorted({feature for feature in grouped if grouped.count(feature) > 1})
    if duplicates:
        raise ValueError(f"feature groups overlap: {duplicates}")
    missing = sorted(set(features) - set(grouped))
    extra = sorted(set(grouped) - set(features))
    if missing or extra:
        raise ValueError(f"feature group coverage mismatch; missing={missing}, extra={extra}")
    assert_feature_names_are_pre_decision(features)


def permutation_attribution(
    model,
    validation: pd.DataFrame,
    features: list[str],
    *,
    repeats: int,
    seed_offset: int,
) -> dict[str, object]:
    """Measure weighted PR-AUC loss after deterministic row permutation."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    validate_feature_groups(features)
    labels = validation["label"]
    weights = _population_weights(labels)
    baseline = float(
        average_precision_score(
            labels,
            model.predict_proba(validation[features])[:, 1],
            sample_weight=weights,
        )
    )

    def evaluate(columns_by_name: dict[str, list[str]], family_offset: int) -> dict[str, object]:
        results: dict[str, object] = {}
        for item_index, (name, columns) in enumerate(columns_by_name.items()):
            scores = []
            for repeat in range(repeats):
                seed = (
                    ATTRIBUTION_SEED
                    + seed_offset * 100_000
                    + family_offset
                    + item_index * 100
                    + repeat
                )
                order = np.random.default_rng(seed).permutation(len(validation))
                permuted = validation[features].copy()
                permuted.loc[:, columns] = permuted[columns].to_numpy()[order]
                scores.append(
                    float(
                        average_precision_score(
                            labels,
                            model.predict_proba(permuted)[:, 1],
                            sample_weight=weights,
                        )
                    )
                )
            results[name] = {
                "columns": columns,
                "permuted_pr_auc_mean": float(np.mean(scores)),
                "permuted_pr_auc_std": float(np.std(scores)),
                "pr_auc_drop_mean": float(baseline - np.mean(scores)),
                "pr_auc_drop_min": float(baseline - np.max(scores)),
                "pr_auc_drop_max": float(baseline - np.min(scores)),
            }
        return results

    feature_columns = {feature: [feature] for feature in features}
    return {
        "baseline_pr_auc": baseline,
        "repeats": repeats,
        "groups": evaluate(FEATURE_GROUPS, 10_000),
        "features": evaluate(feature_columns, 20_000),
    }


def aggregate_importance(
    fold_results: list[dict[str, object]],
    standard_attribution: dict[str, object],
    *,
    result_key: str,
) -> list[dict[str, object]]:
    names = list(standard_attribution[result_key])
    aggregate = []
    for name in names:
        fold_drops = [
            float(fold["attribution"][result_key][name]["pr_auc_drop_mean"])
            for fold in fold_results
        ]
        aggregate.append(
            {
                "name": name,
                "mean_temporal_pr_auc_drop": float(np.mean(fold_drops)),
                "median_temporal_pr_auc_drop": float(np.median(fold_drops)),
                "minimum_temporal_pr_auc_drop": float(np.min(fold_drops)),
                "maximum_temporal_pr_auc_drop": float(np.max(fold_drops)),
                "positive_fold_count": int(sum(value > 0 for value in fold_drops)),
                "fold_count": len(fold_drops),
                "standard_validation_pr_auc_drop": float(
                    standard_attribution[result_key][name]["pr_auc_drop_mean"]
                ),
            }
        )
    aggregate.sort(key=lambda row: row["mean_temporal_pr_auc_drop"], reverse=True)
    for rank, row in enumerate(aggregate, start=1):
        row["rank"] = rank
    return aggregate


def feature_profiles(validation: pd.DataFrame, features: list[str]) -> dict[str, object]:
    profiles: dict[str, object] = {}
    for feature in features:
        class_profiles = {}
        for label, label_name in ((0, "not_bought"), (1, "bought")):
            values = pd.to_numeric(
                validation.loc[validation["label"] == label, feature], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)
            class_profiles[label_name] = {
                "rows": int(len(values)),
                "missing_fraction": float(values.isna().mean()),
                "p10": float(values.quantile(0.1)) if values.notna().any() else None,
                "median": float(values.median()) if values.notna().any() else None,
                "p90": float(values.quantile(0.9)) if values.notna().any() else None,
            }
        profiles[feature] = class_profiles
    return profiles


def run_feature_attribution(dataset_path: Path) -> dict[str, object]:
    frame = pd.read_parquet(dataset_path)
    required = {
        "label",
        "decision_time",
        "token_address",
        "tx_hash",
        "tx_signer",
        "decision_hour_utc",
        "decision_weekday_utc",
        *CREATOR_HISTORY_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if frame["token_address"].duplicated().any():
        raise ValueError("token_address must be unique")
    if frame["label"].nunique() != 2:
        raise ValueError("dataset must contain both classes")

    dataset_sha256 = sha256_file(dataset_path)
    history_manifest_path = dataset_path.parent / "creator_history_manifest.json"
    history_manifest = json.loads(history_manifest_path.read_text(encoding="utf-8"))
    if history_manifest["sha256"] != dataset_sha256:
        raise ValueError("creator-history manifest does not match the attribution dataset")
    if history_manifest["rows"] != len(frame):
        raise ValueError("creator-history manifest row count does not match the dataset")
    if history_manifest["unique_tokens"] != frame["token_address"].nunique():
        raise ValueError("creator-history manifest unique-token count does not match")
    if history_manifest["strict_time_violations"] != 0:
        raise ValueError("creator-history manifest reports strict-time violations")

    canonical_time = pd.to_datetime(frame["decision_time"], utc=True)

    positive_times = frame.loc[frame["label"] == 1, "decision_time"]
    active_start = positive_times.min()
    active_end = positive_times.max()
    active = frame.loc[frame["decision_time"].between(active_start, active_end)].copy()
    outer = chronological_train_validation_test_split(
        active,
        time_column="decision_time",
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    pretest = pd.concat([outer.train, outer.validation], ignore_index=True)
    features = _numeric_features(pretest)
    validate_feature_groups(features)

    folds = expanding_time_folds(pretest, time_column="decision_time")
    fold_results = []
    for fold_number, fold in enumerate(folds, start=1):
        metrics, model = _validation_metrics(fold.train, fold.validation, features)
        attribution = permutation_attribution(
            model,
            fold.validation,
            features,
            repeats=TEMPORAL_REPEATS,
            seed_offset=fold_number,
        )
        fold_results.append({"fold": fold_number, "metrics": metrics, "attribution": attribution})

    standard_metrics, standard_model = _validation_metrics(outer.train, outer.validation, features)
    standard_attribution = permutation_attribution(
        standard_model,
        outer.validation,
        features,
        repeats=STANDARD_REPEATS,
        seed_offset=100,
    )
    group_importance = aggregate_importance(fold_results, standard_attribution, result_key="groups")
    feature_importance = aggregate_importance(
        fold_results, standard_attribution, result_key="features"
    )

    max_evaluated_time = max(
        pd.Timestamp(fold["metrics"]["validation_end"]) for fold in fold_results
    )
    max_evaluated_time = max(max_evaluated_time, outer.validation["decision_time"].max())
    final_holdout_start = outer.test["decision_time"].min()
    if not max_evaluated_time < final_holdout_start:
        raise AssertionError("feature attribution touched the final holdout")

    creator_group = next(row for row in group_importance if row["name"] == "creator_history")
    creator_ranked_first = creator_group["rank"] == 1
    creator_positive_all_folds = creator_group["positive_fold_count"] == creator_group["fold_count"]
    creator_top_all_folds = all(
        max(
            fold["attribution"]["groups"],
            key=lambda name: fold["attribution"]["groups"][name]["pr_auc_drop_mean"],
        )
        == "creator_history"
        for fold in fold_results
    )
    decision = (
        "supported_creator_history_is_dominant"
        if creator_ranked_first and creator_positive_all_folds and creator_top_all_folds
        else "not_supported_creator_history_is_not_consistently_dominant"
    )
    return {
        "experiment": "all_feature_temporal_permutation_attribution",
        "single_hypothesis": (
            "strict_prior_signer_recurrence_and_maturity_dominate_metadata_fee_compute_"
            "transaction_structure_and_clock_features"
        ),
        "decision": decision,
        "predeclared_acceptance_criterion": (
            "strict prior signer history must have positive group permutation PR-AUC drop in "
            "all three expanding folds, rank first by mean temporal drop, and be the largest "
            "group in every fold"
        ),
        "dataset": project_relative(dataset_path),
        "dataset_sha256": dataset_sha256,
        "creator_history_manifest": project_relative(history_manifest_path),
        "creator_history_manifest_sha256": sha256_file(history_manifest_path),
        "code_parent_commit": git_head(),
        "schema": {
            "rows": int(len(frame)),
            "active_rows": int(len(active)),
            "columns": int(len(frame.columns)),
            "unique_tokens": int(frame["token_address"].nunique()),
            "duplicate_tx_hash_rows": int(frame["tx_hash"].duplicated().sum()),
            "positive_rows": int((frame["label"] == 1).sum()),
            "negative_rows": int((frame["label"] == 0).sum()),
            "strict_time_violations": int(history_manifest["strict_time_violations"]),
            "negative_prior_count_rows": int((frame["creator_prior_deploy_count"] < 0).sum()),
            "negative_prior_slot_count_rows": int(
                (frame["creator_prior_active_slot_count"] < 0).sum()
            ),
            "negative_observed_age_rows": int(
                (frame["creator_observed_age_seconds"].dropna() < 0).sum()
            ),
            "negative_recency_rows": int(
                (frame["creator_seconds_since_previous_deploy"].dropna() < 0).sum()
            ),
            "nonzero_count_missing_age_rows": int(
                (
                    frame["creator_prior_deploy_count"].ne(0)
                    & frame["creator_observed_age_seconds"].isna()
                ).sum()
            ),
            "nonzero_count_missing_recency_rows": int(
                (
                    frame["creator_prior_deploy_count"].ne(0)
                    & frame["creator_seconds_since_previous_deploy"].isna()
                ).sum()
            ),
            "utc_hour_mismatch_rows": int(
                frame["decision_hour_utc"].ne(canonical_time.dt.hour).sum()
            ),
            "utc_weekday_mismatch_rows": int(
                frame["decision_weekday_utc"].ne(canonical_time.dt.dayofweek).sum()
            ),
        },
        "history_definition": {
            "creator_key": history_manifest["creator_key"],
            "availability": (
                "all signer-history counts and timestamps aggregate only deployment-index "
                "slots strictly smaller than the current deployment slot"
            ),
            "semantic_limit": (
                "tx_signer is the deployment transaction signer or fee payer and is not proven "
                "to equal the token creator address"
            ),
            "post_deployment_inputs": "none",
        },
        "active_window_start": active_start.isoformat(),
        "active_window_end": active_end.isoformat(),
        "feature_names": features,
        "feature_groups": FEATURE_GROUPS,
        "hyperparameters": BOOSTING_PARAMETERS,
        "temporal_repeats": TEMPORAL_REPEATS,
        "standard_repeats": STANDARD_REPEATS,
        "expanding_folds": fold_results,
        "standard_validation": {
            "metrics": standard_metrics,
            "attribution": standard_attribution,
        },
        "group_importance": group_importance,
        "creator_history_top_group_all_folds": creator_top_all_folds,
        "feature_importance": feature_importance,
        "standard_validation_feature_profiles": feature_profiles(outer.validation, features),
        "max_evaluated_time": max_evaluated_time.isoformat(),
        "final_holdout_start": final_holdout_start.isoformat(),
        "final_holdout_evaluated": False,
        "final_holdout_status": "sealed_no_predictions_generated",
        "interpretation_limit": (
            "Permutation drops are predictive associations, are not additive for correlated "
            "features, and do not establish the target wallet's causal rules."
        ),
    }


def _format_value(value: object) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    numeric = float(value)
    if abs(numeric) >= 1_000_000:
        return f"{numeric:,.0f}"
    if abs(numeric) >= 100:
        return f"{numeric:,.1f}"
    return f"{numeric:.3f}"


def render_report(metrics: dict[str, object]) -> str:
    group_rows = []
    for row in metrics["group_importance"]:
        group_rows.append(
            "| {name} | {mean:.5f} | {minimum:.5f} | {positive}/{folds} | {standard:.5f} |".format(
                name=row["name"].replace("_", " "),
                mean=row["mean_temporal_pr_auc_drop"],
                minimum=row["minimum_temporal_pr_auc_drop"],
                positive=row["positive_fold_count"],
                folds=row["fold_count"],
                standard=row["standard_validation_pr_auc_drop"],
            )
        )

    feature_rows = []
    profiles = metrics["standard_validation_feature_profiles"]
    for row in metrics["feature_importance"][:10]:
        profile = profiles[row["name"]]
        feature_row_template = (
            "| {rank} | {name} | {mean:.5f} | {positive}/{folds} | {bought} | {not_bought} |"
        )
        feature_rows.append(
            feature_row_template.format(
                rank=row["rank"],
                name=FEATURE_LABELS[row["name"]],
                mean=row["mean_temporal_pr_auc_drop"],
                positive=row["positive_fold_count"],
                folds=row["fold_count"],
                bought=_format_value(profile["bought"]["median"]),
                not_bought=_format_value(profile["not_bought"]["median"]),
            )
        )

    creator_rules = []
    importance_by_name = {row["name"]: row for row in metrics["feature_importance"]}
    for feature in CREATOR_HISTORY_FEATURES:
        row = importance_by_name[feature]
        profile = profiles[feature]
        bought = profile["bought"]["median"]
        not_bought = profile["not_bought"]["median"]
        direction = (
            "higher"
            if bought is not None and not_bought is not None and bought > not_bought
            else "lower"
        )
        creator_rules.append(
            f"- **{FEATURE_LABELS[feature]}:** selected tokens have a {direction} validation "
            f"median ({_format_value(bought)} vs {_format_value(not_bought)}); temporal mean "
            f"PR-AUC drop is {row['mean_temporal_pr_auc_drop']:.5f}."
        )

    fold_rows = []
    for fold in metrics["expanding_folds"]:
        creator = fold["attribution"]["groups"]["creator_history"]
        top_group = max(
            fold["attribution"]["groups"],
            key=lambda name: fold["attribution"]["groups"][name]["pr_auc_drop_mean"],
        )
        fold_rows.append(
            "| {fold} | {start} to {end} | {baseline:.5f} | {drop:.5f} | {top} |".format(
                fold=fold["fold"],
                start=pd.Timestamp(fold["metrics"]["validation_start"]).date(),
                end=pd.Timestamp(fold["metrics"]["validation_end"]).date(),
                baseline=fold["attribution"]["baseline_pr_auc"],
                drop=creator["pr_auc_drop_mean"],
                top=top_group.replace("_", " "),
            )
        )

    standard_metrics = metrics["standard_validation"]["metrics"]
    standard_op = standard_metrics["selected_operating_point"]
    standard_rows = [
        f"| PR-AUC | {standard_metrics['population_adjusted_pr_auc']:.5f} |",
        f"| Precision | {standard_op['precision']:.4f} |",
        f"| Recall | {standard_op['recall']:.4f} |",
        f"| F1 | {standard_op['f1']:.4f} |",
        f"| Selected threshold | {standard_op['threshold']:.6f} |",
    ]

    creator_group = next(
        row for row in metrics["group_importance"] if row["name"] == "creator_history"
    )
    transaction_group = next(
        row for row in metrics["group_importance"] if row["name"] == "transaction_structure"
    )
    top_feature = metrics["feature_importance"][0]
    unstable_history = [
        row
        for row in metrics["feature_importance"]
        if row["name"] in CREATOR_HISTORY_FEATURES
        and row["positive_fold_count"] < row["fold_count"]
    ]
    if unstable_history:
        unstable_text = "History features without positive drop in every fold: " + ", ".join(
            FEATURE_LABELS[row["name"]] for row in unstable_history
        )
    else:
        unstable_text = "All four strict history features have positive drops in every fold."
    decision_text = (
        "The hypothesis is supported on development folds: strict prior deployment-signer "
        "history ranks first and has a positive permutation drop in every fold. It is not an "
        "overwhelming lead: the mean drop is "
        f"{creator_group['mean_temporal_pr_auc_drop']:.5f} versus "
        f"{transaction_group['mean_temporal_pr_auc_drop']:.5f} for transaction structure."
        if metrics["decision"] == "supported_creator_history_is_dominant"
        else "The hypothesis is not supported: strict prior deployment-signer history is not "
        "the dominant feature family in every required temporal check."
    )
    group_header = (
        "| Group | Mean temporal PR-AUC drop | Minimum fold drop | Positive folds | "
        "Standard validation drop |"
    )
    return f"""# Leak-free temporal feature attribution

## Decision

{decision_text} This is an interpretability result, not a new performance claim. The final
chronological holdout remains sealed.

{group_header}
|---|---:|---:|---:|---:|
{chr(10).join(group_rows)}

## Temporal checks

| Fold | Validation period | Baseline PR-AUC | Creator-history drop | Largest group |
|---|---|---:|---:|---|
{chr(10).join(fold_rows)}

### Standard validation operating point

| Metric | Value |
|---|---:|
{chr(10).join(standard_rows)}

## Top-10 individual features

The rank is the mean weighted PR-AUC loss across three expanding chronological validation folds.
The sampled class medians are a directional description only; correlated importances are not
additive.

| Rank | Feature | Mean temporal drop | Positive folds | Bought median | Not-bought median |
|---:|---|---:|---:|---:|---:|
{chr(10).join(feature_rows)}

![Temporal permutation importance](figures/feature_attribution.svg)

## Plain-language rule hypothesis

The model is most consistent with two strong, complementary screens: observed deployment-signer
maturity/spacing and the construction of the deployment transaction. It does **not** establish
that either screen is evaluated first. The four strict-history associations are:

{chr(10).join(creator_rules)}

These are associations learned from strictly pre-decision fields. They do not prove the wallet's
implementation or a causal trading rule. The strongest individual diagnostic is
{FEATURE_LABELS[top_feature["name"]]} with mean temporal drop
{top_feature["mean_temporal_pr_auc_drop"]:.5f}. The transaction-structure family remains close to
strict history, so the practical rule hypothesis must retain both. {unstable_text}

## Reproducibility and holdout boundary

- Dataset: `{metrics["dataset"]}`; SHA-256 `{metrics["dataset_sha256"]}`.
- Strict-history manifest: `{metrics["creator_history_manifest"]}`; SHA-256
  `{metrics["creator_history_manifest_sha256"]}`.
- Rows: {metrics["schema"]["rows"]:,}; unique tokens: {metrics["schema"]["unique_tokens"]:,};
  positives: {metrics["schema"]["positive_rows"]:,};
  negatives: {metrics["schema"]["negative_rows"]:,}.
- Strict-history violations: {metrics["schema"]["strict_time_violations"]}; UTC feature
  mismatches:
  {metrics["schema"]["utc_hour_mismatch_rows"] + metrics["schema"]["utc_weekday_mismatch_rows"]}.
- Temporal permutations: {metrics["temporal_repeats"]} deterministic repeats per feature/group;
  standard validation: {metrics["standard_repeats"]} repeats.
- Two complete experiment runs matched the metrics dictionary exactly.
- Same-slot and future signer history are excluded; no later trades, candles, prices, labels,
  realized P&L, or outcome fields enter the model.
- Maximum evaluated time: `{metrics["max_evaluated_time"]}`.
- Final holdout starts: `{metrics["final_holdout_start"]}` and was not predicted.
- Code parent: `{metrics["code_parent_commit"]}`.
"""


def render_svg(metrics: dict[str, object]) -> str:
    rows = metrics["feature_importance"][:10]
    width = 900
    height = 92 + len(rows) * 36
    label_x = 28
    bar_x = 280
    bar_width = 500
    maximum = max(max(float(row["mean_temporal_pr_auc_drop"]), 0.0) for row in rows) or 1.0
    svg_header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    elements = [
        svg_header,
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Arial" font-size="22" font-weight="700" '
        'fill="#111827">Temporal permutation importance</text>',
        '<text x="28" y="58" font-family="Arial" font-size="13" fill="#4b5563">'
        "Mean population-weighted PR-AUC drop across three chronological folds</text>",
    ]
    for index, row in enumerate(rows):
        y = 84 + index * 36
        value = float(row["mean_temporal_pr_auc_drop"])
        rendered_width = max(0.0, value) / maximum * bar_width
        color = "#0f766e" if row["name"] in CREATOR_HISTORY_FEATURES else "#4f46e5"
        label = FEATURE_LABELS[row["name"]]
        elements.extend(
            [
                f'<text x="{label_x}" y="{y + 17}" font-family="Arial" font-size="12" '
                f'fill="#111827">{label}</text>',
                f'<rect x="{bar_x}" y="{y}" width="{rendered_width:.2f}" height="22" '
                f'rx="3" fill="{color}"/>',
                f'<text x="{bar_x + rendered_width + 8:.2f}" y="{y + 16}" '
                f'font-family="Arial" font-size="12" fill="#111827">{value:.5f}</text>',
            ]
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def main() -> None:
    dataset_path = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
    metrics = run_feature_attribution(dataset_path)
    reproduced = run_feature_attribution(dataset_path)
    if reproduced != metrics:
        raise AssertionError("deterministic feature-attribution rerun did not match")
    metrics["reproduction_verification"] = {
        "verified": True,
        "run_count": 2,
        "comparison": "complete_metrics_dictionary_exact_match",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_root = REPORT_DIR.parent
    figure_dir = report_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "feature_attribution.json"
    report_path = report_root / "feature_attribution.md"
    figure_path = figure_dir / "feature_attribution.svg"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(metrics), encoding="utf-8")
    figure_path.write_text(render_svg(metrics), encoding="utf-8")
    append_experiment(
        {
            **metrics,
            "metrics_path": project_relative(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "report_path": project_relative(report_path),
            "report_sha256": sha256_file(report_path),
            "figure_path": project_relative(figure_path),
            "figure_sha256": sha256_file(figure_path),
        }
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
