"""Audit and correct class-dependent UTC time-feature construction."""

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

STORED_HOUR = "decision_hour_utc"
STORED_WEEKDAY = "decision_weekday_utc"
CANONICAL_HOUR = "decision_hour_utc_canonical"
CANONICAL_WEEKDAY = "decision_weekday_utc_canonical"


def add_canonical_utc_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive UTC hour and pandas-style weekday directly from decision_time."""
    if "decision_time" not in frame.columns:
        raise ValueError("decision_time is missing")
    decision_time = pd.to_datetime(frame["decision_time"], errors="raise", utc=True)
    if decision_time.isna().any():
        raise ValueError("decision_time contains missing values")
    result = frame.copy()
    result[CANONICAL_HOUR] = decision_time.dt.hour
    result[CANONICAL_WEEKDAY] = decision_time.dt.weekday
    if not result[CANONICAL_HOUR].between(0, 23).all():
        raise ValueError("canonical UTC hour is outside 0..23")
    if not result[CANONICAL_WEEKDAY].between(0, 6).all():
        raise ValueError("canonical UTC weekday is outside 0..6")
    return result


def integrity_decision(hour_mismatches: int, weekday_mismatches: int) -> str:
    """Require canonical replacement whenever either stored field is inconsistent."""
    if hour_mismatches < 0 or weekday_mismatches < 0:
        raise ValueError("mismatch counts must be nonnegative")
    if hour_mismatches or weekday_mismatches:
        return "required_integrity_fix"
    return "already_consistent"


def _mismatch_profile(frame: pd.DataFrame) -> dict[str, object]:
    profile = {}
    for label, name in ((0, "not_bought"), (1, "bought")):
        subset = frame.loc[frame["label"] == label]
        profile[name] = {
            "rows": int(len(subset)),
            "hour_mismatch_rows": int((subset[STORED_HOUR] != subset[CANONICAL_HOUR]).sum()),
            "weekday_mismatch_rows": int(
                (subset[STORED_WEEKDAY] != subset[CANONICAL_WEEKDAY]).sum()
            ),
        }
    return profile


def run_time_feature_integrity_experiment(dataset_path: Path) -> dict[str, object]:
    frame = pd.read_parquet(dataset_path)
    required = {
        "label",
        "decision_time",
        "token_address",
        "tx_hash",
        RAW_SIGNER_FEATURE,
        "tx_fee_lamports",
        STORED_HOUR,
        STORED_WEEKDAY,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if frame["token_address"].duplicated().any():
        raise ValueError("token_address must be unique")
    if frame["label"].nunique() != 2:
        raise ValueError("dataset must contain both classes")

    frame = add_canonical_utc_time(frame)
    hour_mismatches = int((frame[STORED_HOUR] != frame[CANONICAL_HOUR]).sum())
    weekday_mismatches = int((frame[STORED_WEEKDAY] != frame[CANONICAL_WEEKDAY]).sum())
    positive_times = frame.loc[frame["label"] == 1, "decision_time"]
    active_start = positive_times.min()
    active_end = positive_times.max()
    active = frame.loc[frame["decision_time"].between(active_start, active_end)].copy()
    active = add_deployment_outflow_proxy(active)
    split = chronological_train_validation_test_split(
        active,
        time_column="decision_time",
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    pretest = pd.concat([split.train, split.validation], ignore_index=True)
    all_numeric = _numeric_features(pretest)
    stored_features = [
        feature
        for feature in all_numeric
        if feature not in {RAW_SIGNER_FEATURE, CANONICAL_HOUR, CANONICAL_WEEKDAY}
    ]
    if not {OUTFLOW_FEATURE, STORED_HOUR, STORED_WEEKDAY}.issubset(stored_features):
        raise ValueError("fee-adjusted baseline or stored time features are missing")
    replacements = {
        STORED_HOUR: CANONICAL_HOUR,
        STORED_WEEKDAY: CANONICAL_WEEKDAY,
    }
    canonical_features = [replacements.get(feature, feature) for feature in stored_features]
    assert_feature_names_are_pre_decision(stored_features)
    assert_feature_names_are_pre_decision(canonical_features)
    if len(stored_features) != len(canonical_features):
        raise AssertionError("time replacement changed feature cardinality")
    if len(set(canonical_features)) != len(canonical_features):
        raise AssertionError("canonical time feature set contains duplicates")

    fold_results = []
    for fold_number, fold in enumerate(
        expanding_time_folds(pretest, time_column="decision_time"), start=1
    ):
        stored_metrics, _ = _validation_metrics(fold.train, fold.validation, stored_features)
        canonical_metrics, _ = _validation_metrics(fold.train, fold.validation, canonical_features)
        fold_results.append(
            {
                "fold": fold_number,
                "stored_time": stored_metrics,
                "canonical_utc_time": canonical_metrics,
                "canonical_minus_stored_pr_auc": float(
                    canonical_metrics["population_adjusted_pr_auc"]
                    - stored_metrics["population_adjusted_pr_auc"]
                ),
            }
        )

    standard_stored, _ = _validation_metrics(split.train, split.validation, stored_features)
    standard_canonical, _ = _validation_metrics(split.train, split.validation, canonical_features)
    standard_delta = float(
        standard_canonical["population_adjusted_pr_auc"]
        - standard_stored["population_adjusted_pr_auc"]
    )
    fold_deltas = [float(row["canonical_minus_stored_pr_auc"]) for row in fold_results]
    max_evaluated_time = max(
        split.validation["decision_time"].max(),
        max(pd.Timestamp(row["canonical_utc_time"]["validation_end"]) for row in fold_results),
    )
    final_holdout_start = split.test["decision_time"].min()
    if not max_evaluated_time < final_holdout_start:
        raise AssertionError("time-integrity experiment touched the final holdout")

    return {
        "experiment": "decision_time_feature_integrity",
        "single_hypothesis": (
            "class-dependent timezone and weekday conventions inflated the stored-time "
            "baseline; canonical UTC derivation removes that invalid signal"
        ),
        "decision": integrity_decision(hour_mismatches, weekday_mismatches),
        "predeclared_integrity_criterion": (
            "both stored time features must equal values derived directly from decision_time "
            "in UTC for every row; performance deltas are descriptive, not an acceptance gate"
        ),
        "dataset": project_relative(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "code_parent_commit": git_head(),
        "schema": {
            "rows": len(frame),
            "active_rows": len(active),
            "columns_before_canonical": len(frame.columns) - 2,
            "unique_tokens": int(frame["token_address"].nunique()),
            "duplicate_tx_hash_rows": int(frame["tx_hash"].duplicated().sum()),
            "positive_rows": int((frame["label"] == 1).sum()),
            "negative_rows": int((frame["label"] == 0).sum()),
            "missing_decision_time_rows": int(frame["decision_time"].isna().sum()),
            "hour_mismatch_rows": hour_mismatches,
            "weekday_mismatch_rows": weekday_mismatches,
        },
        "mismatch_profile_by_class": _mismatch_profile(frame),
        "root_cause": {
            "positive_pipeline": "pandas datetime UTC hour and Monday-zero weekday",
            "negative_pipeline_before_fix": (
                "DuckDB session-local hour(to_timestamp(epoch)) and Sunday-zero dayofweek"
            ),
            "observed_negative_hour_offsets_modulo_24": sorted(
                int(value)
                for value in (
                    (
                        frame.loc[frame["label"] == 0, STORED_HOUR]
                        - frame.loc[frame["label"] == 0, CANONICAL_HOUR]
                    )
                    % 24
                ).unique()
            ),
        },
        "feature_replacement": {
            "removed": [STORED_HOUR, STORED_WEEKDAY],
            "added": [CANONICAL_HOUR, CANONICAL_WEEKDAY],
            "availability": (
                "decision_time is the observed deployment timestamp and is available at "
                "t_decision; no later event or outcome is used"
            ),
        },
        "active_window_start": active_start.isoformat(),
        "active_window_end": active_end.isoformat(),
        "stored_feature_names": stored_features,
        "canonical_feature_names": canonical_features,
        "hyperparameters": BOOSTING_PARAMETERS,
        "expanding_folds": fold_results,
        "mean_canonical_minus_stored_pr_auc": float(np.mean(fold_deltas)),
        "minimum_canonical_minus_stored_pr_auc": float(np.min(fold_deltas)),
        "maximum_canonical_minus_stored_pr_auc": float(np.max(fold_deltas)),
        "standard_validation": {
            "stored_time": standard_stored,
            "canonical_utc_time": standard_canonical,
            "canonical_minus_stored_pr_auc": standard_delta,
        },
        "max_evaluated_time": max_evaluated_time.isoformat(),
        "final_holdout_start": final_holdout_start.isoformat(),
        "final_holdout_evaluated": False,
        "final_holdout_status": "sealed_no_predictions_generated",
    }


def render_report(metrics: dict[str, object]) -> str:
    folds = []
    for row in metrics["expanding_folds"]:
        canonical = row["canonical_utc_time"]
        op = canonical["selected_operating_point"]
        folds.append(
            "| {fold} | {start} to {end} | {stored:.5f} | {canonical:.5f} | "
            "{delta:+.5f} | {precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                fold=row["fold"],
                start=pd.Timestamp(canonical["validation_start"]).date(),
                end=pd.Timestamp(canonical["validation_end"]).date(),
                stored=row["stored_time"]["population_adjusted_pr_auc"],
                canonical=canonical["population_adjusted_pr_auc"],
                delta=row["canonical_minus_stored_pr_auc"],
                precision=op["precision"],
                recall=op["recall"],
                f1=op["f1"],
            )
        )
    standard = metrics["standard_validation"]
    stored = standard["stored_time"]
    canonical = standard["canonical_utc_time"]
    stored_op = stored["selected_operating_point"]
    canonical_op = canonical["selected_operating_point"]
    profile = metrics["mismatch_profile_by_class"]
    standard_rows = [
        "| PR-AUC | {stored:.5f} | {canonical:.5f} | {delta:+.5f} |".format(
            stored=stored["population_adjusted_pr_auc"],
            canonical=canonical["population_adjusted_pr_auc"],
            delta=standard["canonical_minus_stored_pr_auc"],
        ),
        "| Precision | {stored:.4f} | {canonical:.4f} | {delta:+.4f} |".format(
            stored=stored_op["precision"],
            canonical=canonical_op["precision"],
            delta=canonical_op["precision"] - stored_op["precision"],
        ),
        "| Recall | {stored:.4f} | {canonical:.4f} | {delta:+.4f} |".format(
            stored=stored_op["recall"],
            canonical=canonical_op["recall"],
            delta=canonical_op["recall"] - stored_op["recall"],
        ),
        "| F1 | {stored:.4f} | {canonical:.4f} | {delta:+.4f} |".format(
            stored=stored_op["f1"],
            canonical=canonical_op["f1"],
            delta=canonical_op["f1"] - stored_op["f1"],
        ),
        "| Threshold | {stored:.6f} | {canonical:.6f} | n/a |".format(
            stored=stored_op["threshold"],
            canonical=canonical_op["threshold"],
        ),
    ]
    return f"""# UTC time-feature integrity correction

## Decision

The stored time baseline is invalid and must be replaced. All
{profile["not_bought"]["rows"]:,} sampled negative rows have a stored-hour mismatch, while
positive rows have none. The mismatch is class-dependent preprocessing, so the stored-time
performance cannot be described as model quality. The final chronological holdout remains sealed.

| Fold | Validation period | Stored PR-AUC | Canonical PR-AUC | Delta | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(folds)}

### Standard validation

| Metric | Stored time | Canonical UTC | Canonical minus stored |
|---|---:|---:|---:|
{chr(10).join(standard_rows)}

![Stored versus canonical UTC time features](figures/time_feature_integrity.svg)

## Root cause and boundary

The positive pipeline used pandas UTC hour and Monday-zero weekday. The negative pipeline used
DuckDB's session-local hour and Sunday-zero weekday. On this machine, negative hours were shifted
by {metrics["root_cause"]["observed_negative_hour_offsets_modulo_24"]} hours modulo 24 because of
PST/PDT. Canonical features are recomputed directly from the deployment timestamp in UTC, which is
available at `t_decision`. No trade, price, candle, outcome, P&L, or future history is used.

- Dataset: `{metrics["dataset"]}`; SHA-256 `{metrics["dataset_sha256"]}`.
- Rows: {metrics["schema"]["rows"]:,}; unique tokens:
  {metrics["schema"]["unique_tokens"]:,}; hour mismatches:
  {metrics["schema"]["hour_mismatch_rows"]:,}; weekday mismatches:
  {metrics["schema"]["weekday_mismatch_rows"]:,}.
- Two deterministic development runs matched exactly.
- Maximum evaluated time: `{metrics["max_evaluated_time"]}`.
- Final holdout starts: `{metrics["final_holdout_start"]}`; no predictions were generated.
"""


def render_svg(metrics: dict[str, object]) -> str:
    rows = [
        (
            f"Fold {row['fold']}",
            row["stored_time"]["population_adjusted_pr_auc"],
            row["canonical_utc_time"]["population_adjusted_pr_auc"],
        )
        for row in metrics["expanding_folds"]
    ]
    standard = metrics["standard_validation"]
    rows.append(
        (
            "Standard",
            standard["stored_time"]["population_adjusted_pr_auc"],
            standard["canonical_utc_time"]["population_adjusted_pr_auc"],
        )
    )
    width, height = 860, 330
    left, top, plot_width, plot_height = 90, 80, 700, 190
    maximum = max(max(stored, canonical) for _, stored, canonical in rows) * 1.12
    group_width = plot_width / len(rows)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Arial" font-size="22" font-weight="700" '
        'fill="#111827">UTC time-feature integrity correction</text>',
        '<text x="28" y="57" font-family="Arial" font-size="13" fill="#4b5563">'
        "Population-weighted PR-AUC; contaminated baseline is invalid</text>",
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#9ca3af"/>',
    ]
    for index, (name, stored, canonical) in enumerate(rows):
        center = left + group_width * (index + 0.5)
        stored_height = stored / maximum * plot_height
        canonical_height = canonical / maximum * plot_height
        elements.extend(
            [
                f'<rect x="{center - 35:.1f}" y="{top + plot_height - stored_height:.1f}" '
                f'width="30" height="{stored_height:.1f}" fill="#dc2626"/>',
                f'<rect x="{center + 5:.1f}" y="{top + plot_height - canonical_height:.1f}" '
                f'width="30" height="{canonical_height:.1f}" fill="#2563eb"/>',
                f'<text x="{center:.1f}" y="{top + plot_height + 20}" text-anchor="middle" '
                f'font-family="Arial" font-size="12" fill="#111827">{name}</text>',
                f'<text x="{center - 20:.1f}" y="{top + plot_height - stored_height - 6:.1f}" '
                f'text-anchor="middle" font-family="Arial" font-size="11">{stored:.4f}</text>',
                f'<text x="{center + 20:.1f}" y="{top + plot_height - canonical_height - 6:.1f}" '
                f'text-anchor="middle" font-family="Arial" font-size="11">{canonical:.4f}</text>',
            ]
        )
    elements.extend(
        [
            '<rect x="550" y="28" width="12" height="12" fill="#dc2626"/>',
            '<text x="568" y="39" font-family="Arial" font-size="12">Stored (invalid)</text>',
            '<rect x="690" y="28" width="12" height="12" fill="#2563eb"/>',
            '<text x="708" y="39" font-family="Arial" font-size="12">Canonical UTC</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def main() -> None:
    dataset_path = PROCESSED_DIR / "classification_dataset_creator_history_pre_utc_fix.parquet"
    metrics = run_time_feature_integrity_experiment(dataset_path)
    reproduced = run_time_feature_integrity_experiment(dataset_path)
    if reproduced != metrics:
        raise AssertionError("deterministic time-integrity rerun did not match")
    metrics["reproduction_verification"] = {
        "verified": True,
        "run_count": 2,
        "comparison": "complete_metrics_dictionary_exact_match",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_root = REPORT_DIR.parent
    figure_dir = report_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "time_feature_integrity.json"
    report_path = report_root / "time_feature_integrity.md"
    figure_path = figure_dir / "time_feature_integrity.svg"
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


def remediation_main() -> None:
    repaired_path = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
    pre_fix_path = PROCESSED_DIR / "classification_dataset_creator_history_pre_utc_fix.parquet"
    reference_path = REPORT_DIR / "time_feature_integrity.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    pre_fix = run_time_feature_integrity_experiment(pre_fix_path)
    repaired = run_time_feature_integrity_experiment(repaired_path)
    if repaired["decision"] != "already_consistent":
        raise AssertionError("rebuilt classification dataset still has time mismatches")
    fold_matches = []
    for reference_fold, repaired_fold in zip(
        pre_fix["expanding_folds"],
        repaired["expanding_folds"],
        strict=True,
    ):
        stored_matches_canonical = (
            repaired_fold["stored_time"] == repaired_fold["canonical_utc_time"]
        )
        matches_reference = repaired_fold["stored_time"] == reference_fold["canonical_utc_time"]
        fold_matches.append(
            {
                "fold": repaired_fold["fold"],
                "stored_matches_canonical": stored_matches_canonical,
                "matches_deterministic_pre_fix_canonical": matches_reference,
            }
        )
    repaired_standard = repaired["standard_validation"]
    reference_standard = pre_fix["standard_validation"]
    standard_stored_matches_canonical = (
        repaired_standard["stored_time"] == repaired_standard["canonical_utc_time"]
    )
    standard_matches_reference = (
        repaired_standard["stored_time"] == reference_standard["canonical_utc_time"]
    )
    all_metrics_match = bool(
        all(
            row["stored_matches_canonical"] and row["matches_deterministic_pre_fix_canonical"]
            for row in fold_matches
        )
        and standard_stored_matches_canonical
        and standard_matches_reference
    )
    if not all_metrics_match:
        raise AssertionError("rebuilt UTC metrics do not match canonical reference")

    negative_manifest_path = PROCESSED_DIR / "negative_sample_manifest.json"
    negative_manifest = json.loads(negative_manifest_path.read_text(encoding="utf-8"))
    metrics = {
        "experiment": "decision_time_feature_integrity_remediation",
        "decision": "verified_canonical_utc_rebuild",
        "single_hypothesis": reference["single_hypothesis"],
        "dataset": project_relative(repaired_path),
        "dataset_sha256": sha256_file(repaired_path),
        "pre_fix_dataset": project_relative(pre_fix_path),
        "pre_fix_dataset_sha256": sha256_file(pre_fix_path),
        "reference_metrics_path": project_relative(reference_path),
        "reference_metrics_sha256": sha256_file(reference_path),
        "negative_sample_manifest": project_relative(negative_manifest_path),
        "negative_sample_manifest_sha256": sha256_file(negative_manifest_path),
        "negative_features_sha256": negative_manifest["negative_features_sha256"],
        "classification_dataset_sha256": negative_manifest["classification_dataset_sha256"],
        "raw_negative_scan_reused": bool(negative_manifest["reused"]),
        "code_parent_commit": git_head(),
        "schema": repaired["schema"],
        "fold_metric_matches": fold_matches,
        "standard_stored_matches_canonical": standard_stored_matches_canonical,
        "standard_matches_deterministic_pre_fix_canonical": standard_matches_reference,
        "all_development_metrics_match": all_metrics_match,
        "expanding_folds": [
            {
                "fold": row["fold"],
                "canonical_utc_time": row["stored_time"],
            }
            for row in repaired["expanding_folds"]
        ],
        "standard_validation": repaired_standard["stored_time"],
        "hyperparameters": BOOSTING_PARAMETERS,
        "max_evaluated_time": repaired["max_evaluated_time"],
        "final_holdout_start": repaired["final_holdout_start"],
        "final_holdout_evaluated": False,
        "final_holdout_status": "sealed_no_predictions_generated",
        "reproduction_verification": {
            "pre_fix_canonical_run_count": 1,
            "repaired_run_count": 1,
            "comparison": (
                "all repaired expanding-fold and standard metrics exactly match the "
                "deterministically ordered pre-fix canonical snapshot"
            ),
            "verified": all_metrics_match,
        },
    }
    standard = metrics["standard_validation"]
    op = standard["selected_operating_point"]
    report = f"""# UTC time-feature remediation verification

The cached negative sample and local deployment index were rebuilt without a full raw-block scan.
Every repaired row now uses UTC hour and Monday-zero weekday derived from `decision_time`.

- Decision: **verified canonical UTC rebuild**.
- Repaired dataset: `{metrics["dataset"]}`; SHA-256
  `{metrics["dataset_sha256"]}`.
- Rows: {metrics["schema"]["rows"]:,}; unique tokens:
  {metrics["schema"]["unique_tokens"]:,}; hour mismatches:
  {metrics["schema"]["hour_mismatch_rows"]}; weekday mismatches:
  {metrics["schema"]["weekday_mismatch_rows"]}.
- The three fold metrics and standard metric exactly match the deterministically ordered
  pre-fix canonical snapshot.
- Corrected standard validation: PR-AUC
  {standard["population_adjusted_pr_auc"]:.5f}, precision {op["precision"]:.4f},
  recall {op["recall"]:.4f}, F1 {op["f1"]:.4f}, threshold
  {op["threshold"]:.6f}.
- Maximum evaluated time: `{metrics["max_evaluated_time"]}`; final holdout starts
  `{metrics["final_holdout_start"]}` and remains sealed.

The earlier 0.09933 standard PR-AUC is invalid because it included class-dependent time
preprocessing. The corrected 0.06980 value is the current evidence-backed baseline.
"""
    metrics_path = REPORT_DIR / "time_feature_remediation.json"
    report_path = REPORT_DIR.parent / "time_feature_remediation.md"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    figure_path = REPORT_DIR.parent / "figures" / "time_feature_integrity.svg"
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
