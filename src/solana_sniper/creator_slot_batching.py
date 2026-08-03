"""Test strict prior creator deployments per active slot as a pre-decision feature."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from solana_sniper.boosting import BOOSTING_PARAMETERS
from solana_sniper.creator_history_stability import (
    _numeric_features,
    _validation_metrics,
    expanding_time_folds,
)
from solana_sniper.deployment_outflow_proxy import (
    PROXY_FEATURE as OUTFLOW_FEATURE,
)
from solana_sniper.deployment_outflow_proxy import (
    RAW_FEATURE as RAW_SIGNER_FEATURE,
)
from solana_sniper.deployment_outflow_proxy import add_deployment_outflow_proxy
from solana_sniper.guardrails import assert_feature_names_are_pre_decision
from solana_sniper.manifest import append_experiment, git_head, sha256_file
from solana_sniper.paths import PROCESSED_DIR, REPORT_DIR, project_relative
from solana_sniper.splits import chronological_train_validation_test_split

COUNT_FEATURE = "creator_prior_deploy_count"
ACTIVE_SLOT_FEATURE = "creator_prior_active_slot_count"
BATCH_FEATURE = "creator_prior_deploys_per_active_slot"


def add_creator_slot_batching(frame: pd.DataFrame) -> pd.DataFrame:
    """Add strict prior deployments per active slot without mutating the input."""
    required = {COUNT_FEATURE, ACTIVE_SLOT_FEATURE}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"creator slot-batching inputs are missing: {missing}")
    count = pd.to_numeric(frame[COUNT_FEATURE], errors="raise")
    active_slots = pd.to_numeric(frame[ACTIVE_SLOT_FEATURE], errors="raise")
    if count.isna().any() or active_slots.isna().any():
        raise ValueError("creator history count inputs contain missing values")
    if (count < 0).any() or (active_slots < 0).any():
        raise ValueError("creator history count inputs must be nonnegative")
    if (active_slots > count).any():
        raise ValueError("creator prior active slot count cannot exceed deploy count")
    if (count.eq(0) != active_slots.eq(0)).any():
        raise ValueError(
            "creator prior deploy and active slot counts must both be zero or positive"
        )

    result = frame.copy()
    denominator = active_slots.where(active_slots.gt(0), 1.0)
    result[BATCH_FEATURE] = np.where(count.eq(0), 0.0, count / denominator)
    if not np.isfinite(result[BATCH_FEATURE]).all():
        raise ValueError("creator slot batching contains nonfinite values")
    if (result[BATCH_FEATURE] < 0).any():
        raise ValueError("creator slot batching must be nonnegative")
    if (result.loc[count.gt(0), BATCH_FEATURE] < 1).any():
        raise ValueError("creator slot batching must be at least one with positive history")
    return result


def batch_decision(fold_deltas: list[float], standard_delta: float) -> str:
    """Apply the predeclared all-development-check improvement criterion."""
    if not fold_deltas:
        raise ValueError("at least one temporal delta is required")
    if all(delta > 0 for delta in [*fold_deltas, standard_delta]):
        return "supported_improves_all_development_checks"
    return "rejected_not_consistent_across_development_checks"


def _class_profile(frame: pd.DataFrame, column: str) -> dict[str, object]:
    profiles = {}
    for label, name in ((0, "not_bought"), (1, "bought")):
        values = pd.to_numeric(frame.loc[frame["label"] == label, column], errors="raise")
        profiles[name] = {
            "rows": len(values),
            "zero_fraction": float((values == 0).mean()),
            "p10": float(values.quantile(0.1)),
            "median": float(values.median()),
            "p90": float(values.quantile(0.9)),
            "p99": float(values.quantile(0.99)),
        }
    return profiles


def run_creator_slot_batching_experiment(dataset_path: Path) -> dict[str, object]:
    frame = pd.read_parquet(dataset_path)
    required = {
        "label",
        "decision_time",
        "token_address",
        "tx_hash",
        RAW_SIGNER_FEATURE,
        "tx_fee_lamports",
        COUNT_FEATURE,
        ACTIVE_SLOT_FEATURE,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if frame["token_address"].duplicated().any():
        raise ValueError("token_address must be unique")
    if frame["label"].nunique() != 2:
        raise ValueError("dataset must contain both classes")

    positive_times = frame.loc[frame["label"] == 1, "decision_time"]
    active_start = positive_times.min()
    active_end = positive_times.max()
    active = frame.loc[frame["decision_time"].between(active_start, active_end)].copy()
    active = add_deployment_outflow_proxy(active)
    active = add_creator_slot_batching(active)
    split = chronological_train_validation_test_split(
        active,
        time_column="decision_time",
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    pretest = pd.concat([split.train, split.validation], ignore_index=True)
    all_numeric = _numeric_features(pretest)
    baseline_features = [
        feature for feature in all_numeric if feature not in {RAW_SIGNER_FEATURE, BATCH_FEATURE}
    ]
    if not {OUTFLOW_FEATURE, COUNT_FEATURE, ACTIVE_SLOT_FEATURE}.issubset(baseline_features):
        raise ValueError("best proxy baseline or creator-history features are missing")
    batch_features = [*baseline_features, BATCH_FEATURE]
    assert_feature_names_are_pre_decision(baseline_features)
    assert_feature_names_are_pre_decision(batch_features)
    if len(batch_features) != len(baseline_features) + 1:
        raise AssertionError("creator slot batching was not the only added feature")
    if len(set(batch_features)) != len(batch_features):
        raise AssertionError("creator slot-batching feature set contains duplicates")

    fold_results = []
    for fold_number, fold in enumerate(
        expanding_time_folds(pretest, time_column="decision_time"), start=1
    ):
        baseline_metrics, _ = _validation_metrics(fold.train, fold.validation, baseline_features)
        batch_metrics, _ = _validation_metrics(fold.train, fold.validation, batch_features)
        fold_results.append(
            {
                "fold": fold_number,
                "baseline": baseline_metrics,
                "with_creator_slot_batching": batch_metrics,
                "batch_minus_baseline_pr_auc": float(
                    batch_metrics["population_adjusted_pr_auc"]
                    - baseline_metrics["population_adjusted_pr_auc"]
                ),
            }
        )

    standard_baseline, _ = _validation_metrics(split.train, split.validation, baseline_features)
    standard_batch, _ = _validation_metrics(split.train, split.validation, batch_features)
    standard_delta = float(
        standard_batch["population_adjusted_pr_auc"]
        - standard_baseline["population_adjusted_pr_auc"]
    )
    fold_deltas = [float(row["batch_minus_baseline_pr_auc"]) for row in fold_results]
    max_evaluated_time = max(
        split.validation["decision_time"].max(),
        max(
            pd.Timestamp(row["with_creator_slot_batching"]["validation_end"])
            for row in fold_results
        ),
    )
    final_holdout_start = split.test["decision_time"].min()
    if not max_evaluated_time < final_holdout_start:
        raise AssertionError("creator slot-batching experiment touched the final holdout")

    validation = add_creator_slot_batching(split.validation)
    return {
        "experiment": "creator_prior_slot_batching",
        "single_hypothesis": (
            "strict prior deployments per historically active deployment slot captures "
            "same-slot batch-mint behavior beyond prior deployment and active-slot counts"
        ),
        "decision": batch_decision(fold_deltas, standard_delta),
        "predeclared_acceptance_criterion": (
            "all three expanding-fold and standard-validation PR-AUC deltas must be "
            "strictly positive"
        ),
        "dataset": project_relative(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "code_parent_commit": git_head(),
        "schema": {
            "rows": len(frame),
            "active_rows": len(active),
            "columns": len(frame.columns),
            "unique_tokens": int(frame["token_address"].nunique()),
            "duplicate_tx_hash_rows": int(frame["tx_hash"].duplicated().sum()),
            "positive_rows": int((frame["label"] == 1).sum()),
            "negative_rows": int((frame["label"] == 0).sum()),
            "missing_count_rows": int(frame[COUNT_FEATURE].isna().sum()),
            "missing_active_slot_rows": int(frame[ACTIVE_SLOT_FEATURE].isna().sum()),
            "active_slots_gt_count_rows": int(
                (frame[ACTIVE_SLOT_FEATURE] > frame[COUNT_FEATURE]).sum()
            ),
            "zero_history_mismatch_rows": int(
                (frame[COUNT_FEATURE].eq(0) != frame[ACTIVE_SLOT_FEATURE].eq(0)).sum()
            ),
            "negative_batch_rows": int((active[BATCH_FEATURE] < 0).sum()),
            "nonfinite_batch_rows": int((~np.isfinite(active[BATCH_FEATURE])).sum()),
        },
        "feature_definition": {
            "name": BATCH_FEATURE,
            "formula": (
                "creator_prior_deploy_count / creator_prior_active_slot_count; "
                "zero when no strictly prior deployment exists"
            ),
            "unit": "strict prior deployments per historically active deployment slot",
            "availability": (
                "both counts aggregate only creator deployment slots strictly before the "
                "current deployment slot and are available at t_decision"
            ),
            "semantic_scope": "historical same-slot batch-mint intensity",
            "interpretation_limit": (
                "the ratio summarizes only deployments present in the supplied deployment "
                "indexes and does not prove wallet ownership or intent"
            ),
        },
        "active_window_start": active_start.isoformat(),
        "active_window_end": active_end.isoformat(),
        "baseline_feature_names": baseline_features,
        "batch_feature_names": batch_features,
        "feature_addition": BATCH_FEATURE,
        "hyperparameters": BOOSTING_PARAMETERS,
        "expanding_folds": fold_results,
        "mean_batch_minus_baseline_pr_auc": float(np.mean(fold_deltas)),
        "minimum_batch_minus_baseline_pr_auc": float(np.min(fold_deltas)),
        "maximum_batch_minus_baseline_pr_auc": float(np.max(fold_deltas)),
        "improved_all_development_checks": bool(
            all(delta > 0 for delta in fold_deltas) and standard_delta > 0
        ),
        "standard_validation": {
            "baseline": standard_baseline,
            "with_creator_slot_batching": standard_batch,
            "batch_minus_baseline_pr_auc": standard_delta,
            "batch_profile": _class_profile(validation, BATCH_FEATURE),
            "count_profile": _class_profile(validation, COUNT_FEATURE),
            "count_to_batch_spearman": float(
                validation[[COUNT_FEATURE, BATCH_FEATURE]].corr(method="spearman").iloc[0, 1]
            ),
            "active_slots_to_batch_spearman": float(
                validation[[ACTIVE_SLOT_FEATURE, BATCH_FEATURE]].corr(method="spearman").iloc[0, 1]
            ),
        },
        "max_evaluated_time": max_evaluated_time.isoformat(),
        "final_holdout_start": final_holdout_start.isoformat(),
        "final_holdout_evaluated": False,
        "final_holdout_status": "sealed_no_predictions_generated",
    }


def render_report(metrics: dict[str, object]) -> str:
    if metrics["decision"] == "supported_improves_all_development_checks":
        decision_text = (
            "The added historical slot-batching feature improves PR-AUC in every "
            "predeclared development check and is retained as a development improvement."
        )
    else:
        decision_text = (
            "The added historical slot-batching feature is rejected because it does not "
            "improve PR-AUC in every predeclared development check."
        )
    fold_rows = []
    for row in metrics["expanding_folds"]:
        model = row["with_creator_slot_batching"]
        op = model["selected_operating_point"]
        fold_rows.append(
            "| {fold} | {start} to {end} | {base:.5f} | {batch:.5f} | {delta:+.5f} | "
            "{precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                fold=row["fold"],
                start=pd.Timestamp(model["validation_start"]).date(),
                end=pd.Timestamp(model["validation_end"]).date(),
                base=row["baseline"]["population_adjusted_pr_auc"],
                batch=model["population_adjusted_pr_auc"],
                delta=row["batch_minus_baseline_pr_auc"],
                precision=op["precision"],
                recall=op["recall"],
                f1=op["f1"],
            )
        )
    standard = metrics["standard_validation"]
    baseline = standard["baseline"]
    batch = standard["with_creator_slot_batching"]
    base_op = baseline["selected_operating_point"]
    batch_op = batch["selected_operating_point"]
    profile = standard["batch_profile"]
    standard_rows = [
        "| PR-AUC | {base:.5f} | {batch:.5f} | {delta:+.5f} |".format(
            base=baseline["population_adjusted_pr_auc"],
            batch=batch["population_adjusted_pr_auc"],
            delta=standard["batch_minus_baseline_pr_auc"],
        ),
        "| Precision | {base:.4f} | {batch:.4f} | {delta:+.4f} |".format(
            base=base_op["precision"],
            batch=batch_op["precision"],
            delta=batch_op["precision"] - base_op["precision"],
        ),
        "| Recall | {base:.4f} | {batch:.4f} | {delta:+.4f} |".format(
            base=base_op["recall"],
            batch=batch_op["recall"],
            delta=batch_op["recall"] - base_op["recall"],
        ),
        "| F1 | {base:.4f} | {batch:.4f} | {delta:+.4f} |".format(
            base=base_op["f1"],
            batch=batch_op["f1"],
            delta=batch_op["f1"] - base_op["f1"],
        ),
        "| Threshold | {base:.6f} | {batch:.6f} | n/a |".format(
            base=base_op["threshold"], batch=batch_op["threshold"]
        ),
    ]
    return f"""# Strict prior creator slot batching

## Decision

{decision_text} This is a two-run-reproduced development result, not an independent final
estimate; the final chronological holdout remains sealed.

| Fold | Validation period | Baseline PR-AUC | Batch PR-AUC | Delta | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(fold_rows)}

### Standard validation operating point

| Metric | Baseline | With slot batching | Slot batching minus baseline |
|---|---:|---:|---:|
{chr(10).join(standard_rows)}

![Baseline versus creator slot batching](figures/creator_slot_batching.svg)

## Interpretation at `t_decision`

The feature divides a creator's strictly prior deployment count by the number of distinct
strictly prior deployment slots. A creator with no earlier indexed deployment receives zero.
Because the current slot is excluded from both window aggregates, no current-sibling or future
deployment, later trade, price, candle, label, or realized P&L enters the model.

On standard validation, bought deployments have median historical deployments per active slot
{profile["bought"]["median"]:.3f} (p90 {profile["bought"]["p90"]:.3f}), versus
{profile["not_bought"]["median"]:.3f} (p90 {profile["not_bought"]["p90"]:.3f}) for sampled
not-bought deployments. Count-to-ratio Spearman correlation is
{standard["count_to_batch_spearman"]:.6f}; active-slots-to-ratio correlation is
{standard["active_slots_to_batch_spearman"]:.6f}. The ratio covers only deployments in the
supplied indexes and does not prove wallet ownership or intent.

## Reproducibility boundary

- Dataset: `{metrics["dataset"]}`; SHA-256 `{metrics["dataset_sha256"]}`.
- Rows: {metrics["schema"]["rows"]:,}; active rows: {metrics["schema"]["active_rows"]:,}; unique
  tokens: {metrics["schema"]["unique_tokens"]:,}; invalid ratio rows:
  {metrics["schema"]["negative_batch_rows"] + metrics["schema"]["nonfinite_batch_rows"]}.
- Exactly one feature is added to the retained fee-adjusted outflow baseline.
- Two complete deterministic runs matched the metrics dictionary exactly.
- Maximum evaluated time: `{metrics["max_evaluated_time"]}`.
- Final holdout starts: `{metrics["final_holdout_start"]}`; no holdout predictions were generated.
"""


def render_svg(metrics: dict[str, object]) -> str:
    rows = [
        (
            f"Fold {row['fold']}",
            row["baseline"]["population_adjusted_pr_auc"],
            row["with_creator_slot_batching"]["population_adjusted_pr_auc"],
        )
        for row in metrics["expanding_folds"]
    ]
    standard = metrics["standard_validation"]
    rows.append(
        (
            "Standard",
            standard["baseline"]["population_adjusted_pr_auc"],
            standard["with_creator_slot_batching"]["population_adjusted_pr_auc"],
        )
    )
    width, height = 860, 330
    left, top, plot_width, plot_height = 90, 80, 700, 190
    maximum = max(max(base, rate) for _, base, rate in rows) * 1.12
    group_width = plot_width / len(rows)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Arial" font-size="22" font-weight="700" '
        'fill="#111827">Strict prior creator slot batching</text>',
        '<text x="28" y="57" font-family="Arial" font-size="13" fill="#4b5563">'
        "Population-weighted PR-AUC; final holdout sealed</text>",
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#9ca3af"/>',
    ]
    for index, (name, base, rate) in enumerate(rows):
        center = left + group_width * (index + 0.5)
        base_height = base / maximum * plot_height
        rate_height = rate / maximum * plot_height
        elements.extend(
            [
                f'<rect x="{center - 35:.1f}" y="{top + plot_height - base_height:.1f}" '
                f'width="30" height="{base_height:.1f}" fill="#64748b"/>',
                f'<rect x="{center + 5:.1f}" y="{top + plot_height - rate_height:.1f}" '
                f'width="30" height="{rate_height:.1f}" fill="#2563eb"/>',
                f'<text x="{center:.1f}" y="{top + plot_height + 20}" text-anchor="middle" '
                f'font-family="Arial" font-size="12" fill="#111827">{name}</text>',
                f'<text x="{center - 20:.1f}" y="{top + plot_height - base_height - 6:.1f}" '
                f'text-anchor="middle" font-family="Arial" font-size="11">{base:.4f}</text>',
                f'<text x="{center + 20:.1f}" y="{top + plot_height - rate_height - 6:.1f}" '
                f'text-anchor="middle" font-family="Arial" font-size="11">{rate:.4f}</text>',
            ]
        )
    elements.extend(
        [
            '<rect x="570" y="28" width="12" height="12" fill="#64748b"/>',
            '<text x="588" y="39" font-family="Arial" font-size="12">Baseline</text>',
            '<rect x="680" y="28" width="12" height="12" fill="#2563eb"/>',
            '<text x="698" y="39" font-family="Arial" font-size="12">With slot batching</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def main() -> None:
    dataset_path = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
    metrics = run_creator_slot_batching_experiment(dataset_path)
    reproduced = run_creator_slot_batching_experiment(dataset_path)
    if reproduced != metrics:
        raise AssertionError("deterministic creator-slot-batching rerun did not match")
    metrics["reproduction_verification"] = {
        "verified": True,
        "run_count": 2,
        "comparison": "complete_metrics_dictionary_exact_match",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_root = REPORT_DIR.parent
    figure_dir = report_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "creator_slot_batching.json"
    report_path = report_root / "creator_slot_batching.md"
    figure_path = figure_dir / "creator_slot_batching.svg"
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
