"""Test creator-history stability and attribution without opening the final holdout."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from solana_sniper.baseline import (
    NON_FEATURE_COLUMNS,
    _best_f1_threshold,
    _population_weights,
)
from solana_sniper.boosting import BOOSTING_PARAMETERS, build_boosting_model
from solana_sniper.guardrails import assert_feature_names_are_pre_decision
from solana_sniper.manifest import append_experiment, git_head, sha256_file
from solana_sniper.paths import PROCESSED_DIR, REPORT_DIR, project_relative
from solana_sniper.splits import chronological_train_validation_test_split

CREATOR_HISTORY_FEATURES = [
    "creator_prior_deploy_count",
    "creator_prior_active_slot_count",
    "creator_observed_age_seconds",
    "creator_seconds_since_previous_deploy",
]
RAW_SIGNER_FEATURE = "signer_lamport_delta"


@dataclass(frozen=True)
class ExpandingFold:
    train: pd.DataFrame
    validation: pd.DataFrame


def expanding_time_folds(
    frame: pd.DataFrame,
    *,
    time_column: str,
    initial_train_fraction: float = 0.4,
    validation_fraction: float = 0.2,
) -> list[ExpandingFold]:
    if not 0 < initial_train_fraction < 1:
        raise ValueError("initial_train_fraction must be between zero and one")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    fold_count = int(round((1 - initial_train_fraction) / validation_fraction))
    if fold_count < 1:
        raise ValueError("fractions do not leave room for validation")
    unique_times = frame[time_column].drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_times) < fold_count + 2:
        raise ValueError("not enough distinct timestamps for expanding folds")
    folds = []
    for fold_index in range(fold_count):
        train_fraction = initial_train_fraction + fold_index * validation_fraction
        validation_end_fraction = min(1.0, train_fraction + validation_fraction)
        train_cut = max(1, int(len(unique_times) * train_fraction))
        train_end = unique_times.iloc[train_cut]
        if validation_end_fraction >= 1:
            validation = frame.loc[frame[time_column] >= train_end].copy()
        else:
            validation_cut = max(train_cut + 1, int(len(unique_times) * validation_end_fraction))
            validation_end = unique_times.iloc[validation_cut]
            validation = frame.loc[
                (frame[time_column] >= train_end) & (frame[time_column] < validation_end)
            ].copy()
        train = frame.loc[frame[time_column] < train_end].copy()
        if train.empty or validation.empty:
            raise ValueError("expanding fold contains an empty partition")
        if not train[time_column].max() < validation[time_column].min():
            raise AssertionError("expanding fold overlaps in time")
        folds.append(ExpandingFold(train=train, validation=validation))
    return folds


def _numeric_features(frame: pd.DataFrame) -> list[str]:
    features = [
        column
        for column in frame.select_dtypes(include=["number", "bool"]).columns
        if column not in NON_FEATURE_COLUMNS
    ]
    assert_feature_names_are_pre_decision(features)
    return features


def stability_decision(fold_deltas: list[float], standard_delta: float) -> str:
    """Apply the predeclared all-development-check improvement criterion."""
    if not fold_deltas:
        raise ValueError("at least one temporal delta is required")
    if all(delta > 0 for delta in [*fold_deltas, standard_delta]):
        return "supported_improves_all_development_checks"
    return "rejected_not_consistent_across_development_checks"


def _class_profiles(frame: pd.DataFrame) -> dict[str, object]:
    profiles = {}
    for feature in CREATOR_HISTORY_FEATURES:
        class_rows = {}
        for label, name in ((0, "not_bought"), (1, "bought")):
            values = pd.to_numeric(frame.loc[frame["label"] == label, feature], errors="raise")
            class_rows[name] = {
                "rows": int(len(values)),
                "missing_fraction": float(values.isna().mean()),
                "median": float(values.median()) if values.notna().any() else None,
                "p10": float(values.quantile(0.1)) if values.notna().any() else None,
                "p90": float(values.quantile(0.9)) if values.notna().any() else None,
            }
        profiles[feature] = class_rows
    return profiles


def _validation_metrics(
    train: pd.DataFrame, validation: pd.DataFrame, features: list[str]
) -> tuple[dict[str, object], object]:
    for name, partition in (("train", train), ("validation", validation)):
        if partition["label"].nunique() != 2:
            raise ValueError(f"{name} partition does not contain both classes")
    model = build_boosting_model()
    model.fit(train[features], train["label"])
    probabilities = model.predict_proba(validation[features])[:, 1]
    weights = _population_weights(validation["label"])
    operating_point = _best_f1_threshold(validation["label"], probabilities, weights)
    return (
        {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "train_end": train["decision_time"].max().isoformat(),
            "validation_start": validation["decision_time"].min().isoformat(),
            "validation_end": validation["decision_time"].max().isoformat(),
            "population_adjusted_pr_auc": float(
                average_precision_score(validation["label"], probabilities, sample_weight=weights)
            ),
            "population_prevalence": float(
                np.average(validation["label"].to_numpy(), weights=weights)
            ),
            "selected_operating_point": operating_point,
        },
        model,
    )


def _permutation_attribution(
    model,
    validation: pd.DataFrame,
    features: list[str],
    *,
    repeats: int = 5,
) -> dict[str, object]:
    labels = validation["label"]
    weights = _population_weights(labels)
    baseline = float(
        average_precision_score(
            labels, model.predict_proba(validation[features])[:, 1], sample_weight=weights
        )
    )
    groups = {"creator_history_family": CREATOR_HISTORY_FEATURES}
    groups.update({feature: [feature] for feature in CREATOR_HISTORY_FEATURES})
    rng = np.random.default_rng(20260801)
    results = {}
    for name, columns in groups.items():
        scores = []
        for _ in range(repeats):
            permuted = validation[features].copy()
            order = rng.permutation(len(permuted))
            permuted.loc[:, columns] = permuted[columns].to_numpy()[order]
            scores.append(
                float(
                    average_precision_score(
                        labels, model.predict_proba(permuted)[:, 1], sample_weight=weights
                    )
                )
            )
        results[name] = {
            "permuted_pr_auc_mean": float(np.mean(scores)),
            "permuted_pr_auc_std": float(np.std(scores)),
            "pr_auc_drop_mean": float(baseline - np.mean(scores)),
        }
    return {"baseline_pr_auc": baseline, "repeats": repeats, "groups": results}


def run_stability(dataset_path: Path) -> dict[str, object]:
    frame = pd.read_parquet(dataset_path)
    required = {
        "label",
        "decision_time",
        "token_address",
        "tx_hash",
        "tx_signer",
        RAW_SIGNER_FEATURE,
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
        raise ValueError("creator-history manifest does not match the experiment dataset")
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
    features_with_creator = _numeric_features(pretest)
    features_without_creator = [
        feature for feature in features_with_creator if feature not in CREATOR_HISTORY_FEATURES
    ]
    if RAW_SIGNER_FEATURE not in features_without_creator:
        raise ValueError("raw deployment-signer balance feature is missing from the baseline")
    if not set(CREATOR_HISTORY_FEATURES).issubset(features_with_creator):
        raise ValueError("strict creator-history feature family is incomplete")
    folds = expanding_time_folds(pretest, time_column="decision_time")
    fold_results = []
    for fold_number, fold in enumerate(folds, start=1):
        baseline_metrics, _ = _validation_metrics(
            fold.train, fold.validation, features_without_creator
        )
        creator_metrics, _ = _validation_metrics(fold.train, fold.validation, features_with_creator)
        fold_results.append(
            {
                "fold": fold_number,
                "without_creator_history": baseline_metrics,
                "with_creator_history": creator_metrics,
                "pr_auc_delta": (
                    creator_metrics["population_adjusted_pr_auc"]
                    - baseline_metrics["population_adjusted_pr_auc"]
                ),
            }
        )
    standard_baseline, _ = _validation_metrics(
        outer.train, outer.validation, features_without_creator
    )
    standard_metrics, standard_model = _validation_metrics(
        outer.train, outer.validation, features_with_creator
    )
    standard_delta = float(
        standard_metrics["population_adjusted_pr_auc"]
        - standard_baseline["population_adjusted_pr_auc"]
    )
    attribution = _permutation_attribution(standard_model, outer.validation, features_with_creator)
    max_evaluated_time = max(
        outer.validation["decision_time"].max(),
        max(pd.Timestamp(fold["with_creator_history"]["validation_end"]) for fold in fold_results),
    )
    final_holdout_start = outer.test["decision_time"].min()
    if not max_evaluated_time < final_holdout_start:
        raise AssertionError("stability analysis touched the final holdout period")
    deltas = [float(fold["pr_auc_delta"]) for fold in fold_results]
    improved_all = all(delta > 0 for delta in deltas) and standard_delta > 0
    return {
        "experiment": "creator_history_temporal_stability_and_attribution",
        "single_hypothesis": (
            "strict prior deployment-signer recurrence maturity and spacing improve "
            "classification across every development-period check after UTC remediation"
        ),
        "decision": stability_decision(deltas, standard_delta),
        "predeclared_acceptance_criterion": (
            "all three expanding-fold and standard-validation PR-AUC deltas must be "
            "strictly positive"
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
                "all counts and timestamps aggregate deployment-index slots strictly smaller "
                "than the current deployment slot; same-slot and future deployments are excluded"
            ),
            "semantic_limit": (
                "tx_signer is the deployment transaction signer or fee payer and is not proven "
                "to equal the token creator address"
            ),
            "post_deployment_inputs": "none",
        },
        "active_window_start": active_start.isoformat(),
        "active_window_end": active_end.isoformat(),
        "hyperparameters": BOOSTING_PARAMETERS,
        "creator_features": CREATOR_HISTORY_FEATURES,
        "baseline_feature_names": features_without_creator,
        "history_feature_names": features_with_creator,
        "expanding_folds": fold_results,
        "positive_delta_fold_count": int(sum(delta > 0 for delta in deltas)),
        "fold_count": len(deltas),
        "mean_pr_auc_delta": float(np.mean(deltas)),
        "minimum_pr_auc_delta": float(np.min(deltas)),
        "maximum_pr_auc_delta": float(np.max(deltas)),
        "improved_all_development_checks": bool(improved_all),
        "standard_validation": {
            "without_creator_history": standard_baseline,
            "with_creator_history": standard_metrics,
            "pr_auc_delta": standard_delta,
            "class_profiles": _class_profiles(outer.validation),
        },
        "permutation_attribution": attribution,
        "max_evaluated_time": max_evaluated_time.isoformat(),
        "final_holdout_start": final_holdout_start.isoformat(),
        "final_holdout_evaluated": False,
        "final_holdout_status": "sealed_no_predictions_generated",
    }


def render_report(metrics: dict[str, object]) -> str:
    if metrics["decision"] == "supported_improves_all_development_checks":
        decision_text = (
            "Retain the strict prior deployment-signer history family. It improves PR-AUC in "
            "every predeclared development check after the UTC remediation."
        )
    else:
        decision_text = (
            "Reject the strict prior deployment-signer history family because it does not "
            "improve PR-AUC in every predeclared development check after UTC remediation."
        )

    fold_rows = []
    for row in metrics["expanding_folds"]:
        with_history = row["with_creator_history"]
        op = with_history["selected_operating_point"]
        fold_rows.append(
            "| {fold} | {start} to {end} | {without:.5f} | {with_history:.5f} | "
            "{delta:+.5f} | {precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                fold=row["fold"],
                start=pd.Timestamp(with_history["validation_start"]).date(),
                end=pd.Timestamp(with_history["validation_end"]).date(),
                without=row["without_creator_history"]["population_adjusted_pr_auc"],
                with_history=with_history["population_adjusted_pr_auc"],
                delta=row["pr_auc_delta"],
                precision=op["precision"],
                recall=op["recall"],
                f1=op["f1"],
            )
        )

    standard = metrics["standard_validation"]
    without = standard["without_creator_history"]
    with_history = standard["with_creator_history"]
    without_op = without["selected_operating_point"]
    with_op = with_history["selected_operating_point"]
    standard_rows = [
        "| PR-AUC | {without:.5f} | {with_history:.5f} | {delta:+.5f} |".format(
            without=without["population_adjusted_pr_auc"],
            with_history=with_history["population_adjusted_pr_auc"],
            delta=standard["pr_auc_delta"],
        ),
        "| Precision | {without:.4f} | {with_history:.4f} | {delta:+.4f} |".format(
            without=without_op["precision"],
            with_history=with_op["precision"],
            delta=with_op["precision"] - without_op["precision"],
        ),
        "| Recall | {without:.4f} | {with_history:.4f} | {delta:+.4f} |".format(
            without=without_op["recall"],
            with_history=with_op["recall"],
            delta=with_op["recall"] - without_op["recall"],
        ),
        "| F1 | {without:.4f} | {with_history:.4f} | {delta:+.4f} |".format(
            without=without_op["f1"],
            with_history=with_op["f1"],
            delta=with_op["f1"] - without_op["f1"],
        ),
        "| Threshold | {without:.6f} | {with_history:.6f} | n/a |".format(
            without=without_op["threshold"], with_history=with_op["threshold"]
        ),
    ]

    attribution_rows = []
    for name, row in metrics["permutation_attribution"]["groups"].items():
        label = (
            "all four history features"
            if name == "creator_history_family"
            else name.replace("creator_", "").replace("_", " ")
        )
        attribution_rows.append(
            f"| {label} | {row['permuted_pr_auc_mean']:.5f} | {row['pr_auc_drop_mean']:+.5f} |"
        )

    profile_rows = []
    for name, row in standard["class_profiles"].items():
        bought = row["bought"]["median"]
        not_bought = row["not_bought"]["median"]
        profile_rows.append(f"| {name} | {bought:,.3f} | {not_bought:,.3f} |")

    invalid_history_rows = sum(
        metrics["schema"][name]
        for name in (
            "strict_time_violations",
            "negative_prior_count_rows",
            "negative_prior_slot_count_rows",
            "negative_observed_age_rows",
            "negative_recency_rows",
            "nonzero_count_missing_age_rows",
            "nonzero_count_missing_recency_rows",
        )
    )
    return f"""# Strict prior deployment-signer history stability

## Decision

{decision_text} The complete metrics dictionary matched across two deterministic runs. This is a
development-period result, not an independent final estimate; the final chronological holdout
remains sealed.

| Fold | Validation period | Without history | With history | Delta | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(fold_rows)}

### Standard validation operating point

| Metric | Without history | With history | History minus baseline |
|---|---:|---:|---:|
{chr(10).join(standard_rows)}

![Strict prior history stability](figures/creator_history_stability.svg)

## Attribution and interpretable rule

Permutation is evaluated only on standard development validation with the final holdout sealed.
Correlated feature drops are not additive.

| Permuted feature or group | Mean permuted PR-AUC | PR-AUC drop |
|---|---:|---:|
{chr(10).join(attribution_rows)}

| Strict prior feature | Bought median | Sampled not-bought median |
|---|---:|---:|
{chr(10).join(profile_rows)}

The supported rule is limited to the observed deployment signer or fee payer: the target prefers
signers with longer observed deployment history and more spacing since their previous deploy,
while recurrence counts provide additional context. `tx_signer` is not proven to equal the token
creator address, so these features must not be described as verified wallet age or creator identity.

## `t_decision` boundary and reproducibility

- Dataset: `{metrics["dataset"]}`; SHA-256 `{metrics["dataset_sha256"]}`.
- Strict-history manifest: `{metrics["creator_history_manifest"]}`; SHA-256
  `{metrics["creator_history_manifest_sha256"]}`.
- Rows: {metrics["schema"]["rows"]:,}; active rows: {metrics["schema"]["active_rows"]:,}; unique
  tokens: {metrics["schema"]["unique_tokens"]:,}; invalid strict-history rows:
  {invalid_history_rows}; UTC time mismatches:
  {metrics["schema"]["utc_hour_mismatch_rows"] + metrics["schema"]["utc_weekday_mismatch_rows"]}.
- Same-slot and future deployments are excluded by strict smaller-slot windows. No later trades,
  candles, prices, labels, realized P&L, or future signer history enter these features.
- Baseline uses the raw deployment-signer balance delta; fee-adjusted nonfee outflow remains a
  semantic interpretation proxy, not a demonstrated classifier improvement.
- Maximum evaluated time: `{metrics["max_evaluated_time"]}`.
- Final holdout starts: `{metrics["final_holdout_start"]}`; no holdout predictions were generated.
- Code parent: `{metrics["code_parent_commit"]}`.
"""


def render_svg(metrics: dict[str, object]) -> str:
    rows = [
        (
            f"Fold {row['fold']}",
            row["without_creator_history"]["population_adjusted_pr_auc"],
            row["with_creator_history"]["population_adjusted_pr_auc"],
        )
        for row in metrics["expanding_folds"]
    ]
    standard = metrics["standard_validation"]
    rows.append(
        (
            "Standard",
            standard["without_creator_history"]["population_adjusted_pr_auc"],
            standard["with_creator_history"]["population_adjusted_pr_auc"],
        )
    )
    width, height = 860, 330
    left, top, plot_width, plot_height = 90, 80, 700, 190
    maximum = max(max(without, with_history) for _, without, with_history in rows) * 1.12
    group_width = plot_width / len(rows)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Arial" font-size="22" font-weight="700" '
        'fill="#111827">Strict prior deployment-signer history</text>',
        '<text x="28" y="57" font-family="Arial" font-size="13" fill="#4b5563">'
        "Population-weighted PR-AUC after UTC remediation; final holdout sealed</text>",
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#9ca3af"/>',
    ]
    for index, (name, without, with_history) in enumerate(rows):
        center = left + group_width * (index + 0.5)
        without_height = without / maximum * plot_height
        with_height = with_history / maximum * plot_height
        elements.extend(
            [
                f'<rect x="{center - 35:.1f}" y="{top + plot_height - without_height:.1f}" '
                f'width="30" height="{without_height:.1f}" fill="#64748b"/>',
                f'<rect x="{center + 5:.1f}" y="{top + plot_height - with_height:.1f}" '
                f'width="30" height="{with_height:.1f}" fill="#2563eb"/>',
                f'<text x="{center:.1f}" y="{top + plot_height + 20}" text-anchor="middle" '
                f'font-family="Arial" font-size="12" fill="#111827">{name}</text>',
                f'<text x="{center - 20:.1f}" '
                f'y="{top + plot_height - without_height - 6:.1f}" text-anchor="middle" '
                f'font-family="Arial" font-size="11">{without:.4f}</text>',
                f'<text x="{center + 20:.1f}" y="{top + plot_height - with_height - 6:.1f}" '
                f'text-anchor="middle" font-family="Arial" font-size="11">'
                f"{with_history:.4f}</text>",
            ]
        )
    elements.extend(
        [
            '<rect x="545" y="28" width="12" height="12" fill="#64748b"/>',
            '<text x="563" y="39" font-family="Arial" font-size="12">Without history</text>',
            '<rect x="680" y="28" width="12" height="12" fill="#2563eb"/>',
            '<text x="698" y="39" font-family="Arial" font-size="12">With history</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def main() -> None:
    dataset_path = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
    metrics = run_stability(dataset_path)
    reproduced = run_stability(dataset_path)
    if reproduced != metrics:
        raise AssertionError("deterministic creator-history stability rerun did not match")
    metrics["reproduction_verification"] = {
        "verified": True,
        "run_count": 2,
        "comparison": "complete_metrics_dictionary_exact_match",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_root = REPORT_DIR.parent
    figure_dir = report_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "creator_history_stability.json"
    report_path = report_root / "creator_history_stability.md"
    figure_path = figure_dir / "creator_history_stability.svg"
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
