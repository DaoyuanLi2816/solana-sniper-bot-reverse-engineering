"""Test priority fee per consumed compute unit as a deployment-urgency feature."""

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

RAW_PRIORITY_FEATURE = "priority_fee_lamports"
INTENSITY_FEATURE = "priority_fee_lamports_per_compute_unit"
COMPUTE_FEATURE = "compute_units"
NONINFERIORITY_MARGIN = 0.002


def add_priority_fee_intensity(frame: pd.DataFrame) -> pd.DataFrame:
    """Add priority-fee lamports per consumed compute unit without mutating input."""
    required = {RAW_PRIORITY_FEATURE, COMPUTE_FEATURE}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"priority intensity inputs are missing: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError("priority intensity inputs contain missing values")
    compute_units = pd.to_numeric(frame[COMPUTE_FEATURE], errors="raise")
    if (compute_units <= 0).any():
        raise ValueError("compute units must be strictly positive")
    result = frame.copy()
    priority_fee = pd.to_numeric(result[RAW_PRIORITY_FEATURE], errors="raise")
    result[INTENSITY_FEATURE] = priority_fee / compute_units
    if not np.isfinite(result[INTENSITY_FEATURE]).all():
        raise ValueError("priority intensity contains nonfinite values")
    if (result[INTENSITY_FEATURE] < 0).any():
        raise ValueError("priority intensity must be nonnegative")
    return result


def intensity_decision(fold_deltas: list[float], standard_delta: float) -> str:
    """Classify the predeclared improvement and noninferiority outcomes."""
    if not fold_deltas:
        raise ValueError("at least one temporal delta is required")
    checks = [*fold_deltas, standard_delta]
    if all(delta > 0 for delta in checks):
        return "supported_improves_all_development_checks"
    if min(checks) >= -NONINFERIORITY_MARGIN:
        return "retained_semantically_within_noninferiority_margin"
    return "rejected_exceeds_noninferiority_margin"


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


def run_priority_intensity_experiment(dataset_path: Path) -> dict[str, object]:
    frame = pd.read_parquet(dataset_path)
    required = {
        "label",
        "decision_time",
        "token_address",
        "tx_hash",
        RAW_SIGNER_FEATURE,
        "tx_fee_lamports",
        RAW_PRIORITY_FEATURE,
        COMPUTE_FEATURE,
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
    active = add_priority_fee_intensity(active)
    split = chronological_train_validation_test_split(
        active,
        time_column="decision_time",
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    pretest = pd.concat([split.train, split.validation], ignore_index=True)
    all_numeric = _numeric_features(pretest)
    baseline_features = [
        feature for feature in all_numeric if feature not in {RAW_SIGNER_FEATURE, INTENSITY_FEATURE}
    ]
    if OUTFLOW_FEATURE not in baseline_features or RAW_PRIORITY_FEATURE not in baseline_features:
        raise ValueError("best proxy baseline features are missing")
    intensity_features = [
        INTENSITY_FEATURE if feature == RAW_PRIORITY_FEATURE else feature
        for feature in baseline_features
    ]
    assert_feature_names_are_pre_decision(baseline_features)
    assert_feature_names_are_pre_decision(intensity_features)
    if len(baseline_features) != len(intensity_features):
        raise AssertionError("single-feature replacement changed feature cardinality")
    if len(set(intensity_features)) != len(intensity_features):
        raise AssertionError("priority intensity feature set contains duplicates")

    fold_results = []
    for fold_number, fold in enumerate(
        expanding_time_folds(pretest, time_column="decision_time"), start=1
    ):
        baseline_metrics, _ = _validation_metrics(fold.train, fold.validation, baseline_features)
        intensity_metrics, _ = _validation_metrics(fold.train, fold.validation, intensity_features)
        fold_results.append(
            {
                "fold": fold_number,
                "absolute_priority_fee": baseline_metrics,
                "priority_fee_intensity": intensity_metrics,
                "intensity_minus_absolute_pr_auc": float(
                    intensity_metrics["population_adjusted_pr_auc"]
                    - baseline_metrics["population_adjusted_pr_auc"]
                ),
            }
        )

    standard_baseline, _ = _validation_metrics(split.train, split.validation, baseline_features)
    standard_intensity, _ = _validation_metrics(split.train, split.validation, intensity_features)
    standard_delta = float(
        standard_intensity["population_adjusted_pr_auc"]
        - standard_baseline["population_adjusted_pr_auc"]
    )
    fold_deltas = [float(row["intensity_minus_absolute_pr_auc"]) for row in fold_results]
    max_evaluated_time = max(
        split.validation["decision_time"].max(),
        max(pd.Timestamp(row["priority_fee_intensity"]["validation_end"]) for row in fold_results),
    )
    final_holdout_start = split.test["decision_time"].min()
    if not max_evaluated_time < final_holdout_start:
        raise AssertionError("priority intensity experiment touched the final holdout")

    return {
        "experiment": "deployment_priority_fee_intensity",
        "single_hypothesis": (
            "priority fee per consumed compute unit is a more stable deployment-urgency "
            "feature than absolute priority fee on the fee-adjusted outflow baseline"
        ),
        "decision": intensity_decision(fold_deltas, standard_delta),
        "predeclared_noninferiority_margin_pr_auc": NONINFERIORITY_MARGIN,
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
            "nonpositive_compute_rows": int((frame[COMPUTE_FEATURE] <= 0).sum()),
        },
        "feature_definition": {
            "name": INTENSITY_FEATURE,
            "formula": "priority_fee_lamports / compute_units",
            "unit": "lamports per consumed compute unit",
            "availability": (
                "deployment transaction fee and compute metadata, available when the "
                "deployment transaction is observed at t_decision"
            ),
            "semantic_scope": "realized priority-fee intensity of the deployment transaction",
            "interpretation_limit": (
                "consumed compute units are not the requested compute limit; intensity is a "
                "realized transaction-level urgency proxy"
            ),
        },
        "active_window_start": active_start.isoformat(),
        "active_window_end": active_end.isoformat(),
        "baseline_feature_names": baseline_features,
        "intensity_feature_names": intensity_features,
        "feature_replacement": {
            "removed": RAW_PRIORITY_FEATURE,
            "added": INTENSITY_FEATURE,
        },
        "hyperparameters": BOOSTING_PARAMETERS,
        "expanding_folds": fold_results,
        "mean_intensity_minus_absolute_pr_auc": float(np.mean(fold_deltas)),
        "minimum_intensity_minus_absolute_pr_auc": float(np.min(fold_deltas)),
        "maximum_intensity_minus_absolute_pr_auc": float(np.max(fold_deltas)),
        "improved_all_development_checks": bool(
            all(delta > 0 for delta in fold_deltas) and standard_delta > 0
        ),
        "standard_validation": {
            "absolute_priority_fee": standard_baseline,
            "priority_fee_intensity": standard_intensity,
            "intensity_minus_absolute_pr_auc": standard_delta,
            "absolute_fee_profile": _class_profile(split.validation, RAW_PRIORITY_FEATURE),
            "intensity_profile": _class_profile(split.validation, INTENSITY_FEATURE),
            "absolute_to_intensity_spearman": float(
                split.validation[[RAW_PRIORITY_FEATURE, INTENSITY_FEATURE]]
                .corr(method="spearman")
                .iloc[0, 1]
            ),
        },
        "max_evaluated_time": max_evaluated_time.isoformat(),
        "final_holdout_start": final_holdout_start.isoformat(),
        "final_holdout_evaluated": False,
        "final_holdout_status": "sealed_no_predictions_generated",
    }


def render_report(metrics: dict[str, object]) -> str:
    decision_text = {
        "supported_improves_all_development_checks": (
            "The intensity replacement improves PR-AUC in every predeclared development check."
        ),
        "retained_semantically_within_noninferiority_margin": (
            "The intensity replacement is retained only as a semantically clearer, "
            "noninferior parameterization; it does not improve every check."
        ),
        "rejected_exceeds_noninferiority_margin": (
            "The intensity replacement is rejected because at least one development check "
            "exceeds the predeclared 0.002 PR-AUC loss margin."
        ),
    }[metrics["decision"]]
    fold_rows = []
    for row in metrics["expanding_folds"]:
        intensity_op = row["priority_fee_intensity"]["selected_operating_point"]
        fold_rows.append(
            "| {fold} | {start} to {end} | {raw:.5f} | {intensity:.5f} | {delta:+.5f} | "
            "{precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                fold=row["fold"],
                start=pd.Timestamp(row["priority_fee_intensity"]["validation_start"]).date(),
                end=pd.Timestamp(row["priority_fee_intensity"]["validation_end"]).date(),
                raw=row["absolute_priority_fee"]["population_adjusted_pr_auc"],
                intensity=row["priority_fee_intensity"]["population_adjusted_pr_auc"],
                delta=row["intensity_minus_absolute_pr_auc"],
                precision=intensity_op["precision"],
                recall=intensity_op["recall"],
                f1=intensity_op["f1"],
            )
        )
    standard = metrics["standard_validation"]
    absolute = standard["absolute_priority_fee"]
    intensity = standard["priority_fee_intensity"]
    absolute_op = absolute["selected_operating_point"]
    intensity_op = intensity["selected_operating_point"]
    bought = standard["intensity_profile"]["bought"]
    not_bought = standard["intensity_profile"]["not_bought"]
    standard_rows = [
        "| PR-AUC | {absolute:.5f} | {intensity:.5f} | {delta:+.5f} |".format(
            absolute=absolute["population_adjusted_pr_auc"],
            intensity=intensity["population_adjusted_pr_auc"],
            delta=standard["intensity_minus_absolute_pr_auc"],
        ),
        "| Precision | {absolute:.4f} | {intensity:.4f} | {delta:+.4f} |".format(
            absolute=absolute_op["precision"],
            intensity=intensity_op["precision"],
            delta=intensity_op["precision"] - absolute_op["precision"],
        ),
        "| Recall | {absolute:.4f} | {intensity:.4f} | {delta:+.4f} |".format(
            absolute=absolute_op["recall"],
            intensity=intensity_op["recall"],
            delta=intensity_op["recall"] - absolute_op["recall"],
        ),
        "| F1 | {absolute:.4f} | {intensity:.4f} | {delta:+.4f} |".format(
            absolute=absolute_op["f1"],
            intensity=intensity_op["f1"],
            delta=intensity_op["f1"] - absolute_op["f1"],
        ),
        "| Threshold | {absolute:.6f} | {intensity:.6f} | n/a |".format(
            absolute=absolute_op["threshold"], intensity=intensity_op["threshold"]
        ),
    ]
    return f"""# Priority-fee intensity at deployment

## Decision

{decision_text} This is a two-run-reproduced development result, not an independent final
estimate; the final chronological holdout remains sealed.

| Fold | Validation period | Absolute PR-AUC | Intensity PR-AUC | Delta | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(fold_rows)}

### Standard validation operating point

| Metric | Absolute priority fee | Fee per compute unit | Intensity minus absolute |
|---|---:|---:|---:|
{chr(10).join(standard_rows)}

![Absolute priority fee versus fee intensity](figures/priority_fee_intensity.svg)

## Interpretation at `t_decision`

The feature is `priority_fee_lamports / compute_units`. Both values come from the observed
deployment transaction. It is therefore available at `t_decision`; no later trade, price, candle,
label, realized P&L, or future deployer history enters the model.

On standard validation, bought deployments have median intensity {bought["median"]:.4f} lamports/CU
(p90 {bought["p90"]:.4f}), versus {not_bought["median"]:.4f} (p90
{not_bought["p90"]:.4f}) for sampled not-bought deployments. The raw-to-intensity Spearman
correlation is {standard["absolute_to_intensity_spearman"]:.6f}. Consumed compute units are not the
requested compute limit, so this is a realized transaction-urgency proxy rather than an exact bid.

## Reproducibility boundary

- Dataset: `{metrics["dataset"]}`; SHA-256 `{metrics["dataset_sha256"]}`.
- Rows: {metrics["schema"]["rows"]:,}; active rows: {metrics["schema"]["active_rows"]:,}; unique
  tokens: {metrics["schema"]["unique_tokens"]:,}; nonpositive compute rows:
  {metrics["schema"]["nonpositive_compute_rows"]}.
- Exactly one feature is replaced on the retained fee-adjusted outflow baseline.
- Two complete deterministic runs matched the metrics dictionary exactly.
- Maximum evaluated time: `{metrics["max_evaluated_time"]}`.
- Final holdout starts: `{metrics["final_holdout_start"]}`; no holdout predictions were generated.
"""


def render_svg(metrics: dict[str, object]) -> str:
    rows = [
        (
            f"Fold {row['fold']}",
            row["absolute_priority_fee"]["population_adjusted_pr_auc"],
            row["priority_fee_intensity"]["population_adjusted_pr_auc"],
        )
        for row in metrics["expanding_folds"]
    ]
    standard = metrics["standard_validation"]
    rows.append(
        (
            "Standard",
            standard["absolute_priority_fee"]["population_adjusted_pr_auc"],
            standard["priority_fee_intensity"]["population_adjusted_pr_auc"],
        )
    )
    width, height = 860, 330
    left, top, plot_width, plot_height = 90, 80, 700, 190
    maximum = max(max(absolute, intensity) for _, absolute, intensity in rows) * 1.12
    group_width = plot_width / len(rows)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Arial" font-size="22" font-weight="700" '
        'fill="#111827">Deployment priority-fee intensity</text>',
        '<text x="28" y="57" font-family="Arial" font-size="13" fill="#4b5563">'
        "Population-weighted PR-AUC; final holdout sealed</text>",
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#9ca3af"/>',
    ]
    for index, (name, absolute, intensity) in enumerate(rows):
        center = left + group_width * (index + 0.5)
        absolute_height = absolute / maximum * plot_height
        intensity_height = intensity / maximum * plot_height
        elements.extend(
            [
                f'<rect x="{center - 35:.1f}" y="{top + plot_height - absolute_height:.1f}" '
                f'width="30" height="{absolute_height:.1f}" fill="#64748b"/>',
                f'<rect x="{center + 5:.1f}" y="{top + plot_height - intensity_height:.1f}" '
                f'width="30" height="{intensity_height:.1f}" fill="#7c3aed"/>',
                f'<text x="{center:.1f}" y="{top + plot_height + 20}" text-anchor="middle" '
                f'font-family="Arial" font-size="12" fill="#111827">{name}</text>',
                f'<text x="{center - 20:.1f}" y="{top + plot_height - absolute_height - 6:.1f}" '
                f'text-anchor="middle" font-family="Arial" font-size="11">{absolute:.4f}</text>',
                f'<text x="{center + 20:.1f}" y="{top + plot_height - intensity_height - 6:.1f}" '
                f'text-anchor="middle" font-family="Arial" font-size="11">{intensity:.4f}</text>',
            ]
        )
    elements.extend(
        [
            '<rect x="570" y="28" width="12" height="12" fill="#64748b"/>',
            '<text x="588" y="39" font-family="Arial" font-size="12">Absolute fee</text>',
            '<rect x="680" y="28" width="12" height="12" fill="#7c3aed"/>',
            '<text x="698" y="39" font-family="Arial" font-size="12">Fee / CU</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def main() -> None:
    dataset_path = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
    metrics = run_priority_intensity_experiment(dataset_path)
    reproduced = run_priority_intensity_experiment(dataset_path)
    if reproduced != metrics:
        raise AssertionError("deterministic priority-intensity rerun did not match")
    metrics["reproduction_verification"] = {
        "verified": True,
        "run_count": 2,
        "comparison": "complete_metrics_dictionary_exact_match",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_root = REPORT_DIR.parent
    figure_dir = report_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "priority_fee_intensity.json"
    report_path = report_root / "priority_fee_intensity.md"
    figure_path = figure_dir / "priority_fee_intensity.svg"
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
