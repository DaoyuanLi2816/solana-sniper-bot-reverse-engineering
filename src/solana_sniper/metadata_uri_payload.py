"""Test URI payload length as a metadata feature available at deployment."""

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

RAW_URI_FEATURE = "metadata_uri_length"
IPFS_FLAG_FEATURE = "metadata_uri_is_ipfs"
PAYLOAD_FEATURE = "metadata_uri_payload_length"
IPFS_PREFIX_LENGTH = len("ipfs://")
NONINFERIORITY_MARGIN = 0.002


def add_metadata_uri_payload_length(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the fixed ``ipfs://`` prefix from URI length without mutating input."""
    required = {RAW_URI_FEATURE, IPFS_FLAG_FEATURE}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"metadata URI payload inputs are missing: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError("metadata URI payload inputs contain missing values")
    raw_length = pd.to_numeric(frame[RAW_URI_FEATURE], errors="raise")
    ipfs_flag = pd.to_numeric(frame[IPFS_FLAG_FEATURE], errors="raise")
    if (raw_length < 0).any():
        raise ValueError("metadata URI length must be nonnegative")
    if not ipfs_flag.isin([0, 1]).all():
        raise ValueError("metadata URI IPFS flag must be binary")
    result = frame.copy()
    result[PAYLOAD_FEATURE] = raw_length - IPFS_PREFIX_LENGTH * ipfs_flag
    if (result[PAYLOAD_FEATURE] < 0).any():
        raise ValueError("metadata URI payload length must be nonnegative")
    if not np.isfinite(result[PAYLOAD_FEATURE]).all():
        raise ValueError("metadata URI payload length contains nonfinite values")
    return result


def payload_decision(fold_deltas: list[float], standard_delta: float) -> str:
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


def run_metadata_uri_payload_experiment(dataset_path: Path) -> dict[str, object]:
    frame = pd.read_parquet(dataset_path)
    required = {
        "label",
        "decision_time",
        "token_address",
        "tx_hash",
        RAW_SIGNER_FEATURE,
        "tx_fee_lamports",
        RAW_URI_FEATURE,
        IPFS_FLAG_FEATURE,
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
    active = add_metadata_uri_payload_length(active)
    split = chronological_train_validation_test_split(
        active,
        time_column="decision_time",
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    pretest = pd.concat([split.train, split.validation], ignore_index=True)
    all_numeric = _numeric_features(pretest)
    baseline_features = [
        feature for feature in all_numeric if feature not in {RAW_SIGNER_FEATURE, PAYLOAD_FEATURE}
    ]
    if not {OUTFLOW_FEATURE, RAW_URI_FEATURE, IPFS_FLAG_FEATURE}.issubset(baseline_features):
        raise ValueError("best proxy baseline or metadata features are missing")
    payload_features = [
        PAYLOAD_FEATURE if feature == RAW_URI_FEATURE else feature for feature in baseline_features
    ]
    assert_feature_names_are_pre_decision(baseline_features)
    assert_feature_names_are_pre_decision(payload_features)
    if len(baseline_features) != len(payload_features):
        raise AssertionError("single-feature replacement changed feature cardinality")
    if len(set(payload_features)) != len(payload_features):
        raise AssertionError("metadata URI payload feature set contains duplicates")

    fold_results = []
    for fold_number, fold in enumerate(
        expanding_time_folds(pretest, time_column="decision_time"), start=1
    ):
        raw_metrics, _ = _validation_metrics(fold.train, fold.validation, baseline_features)
        payload_metrics, _ = _validation_metrics(fold.train, fold.validation, payload_features)
        fold_results.append(
            {
                "fold": fold_number,
                "raw_uri_length": raw_metrics,
                "uri_payload_length": payload_metrics,
                "payload_minus_raw_pr_auc": float(
                    payload_metrics["population_adjusted_pr_auc"]
                    - raw_metrics["population_adjusted_pr_auc"]
                ),
            }
        )

    standard_raw, _ = _validation_metrics(split.train, split.validation, baseline_features)
    standard_payload, _ = _validation_metrics(split.train, split.validation, payload_features)
    standard_delta = float(
        standard_payload["population_adjusted_pr_auc"] - standard_raw["population_adjusted_pr_auc"]
    )
    fold_deltas = [float(row["payload_minus_raw_pr_auc"]) for row in fold_results]
    max_evaluated_time = max(
        split.validation["decision_time"].max(),
        max(pd.Timestamp(row["uri_payload_length"]["validation_end"]) for row in fold_results),
    )
    final_holdout_start = split.test["decision_time"].min()
    if not max_evaluated_time < final_holdout_start:
        raise AssertionError("metadata URI payload experiment touched the final holdout")

    validation = add_metadata_uri_payload_length(split.validation)
    return {
        "experiment": "metadata_uri_payload_length",
        "single_hypothesis": (
            "IPFS URI payload length is a more stable deployment-metadata feature than raw URI "
            "length because it removes the fixed ipfs:// scheme prefix"
        ),
        "decision": payload_decision(fold_deltas, standard_delta),
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
            "ipfs_flag_values": sorted(int(value) for value in active[IPFS_FLAG_FEATURE].unique()),
            "negative_payload_rows": int((active[PAYLOAD_FEATURE] < 0).sum()),
            "ipfs_shorter_than_prefix_rows": int(
                (
                    (active[IPFS_FLAG_FEATURE] == 1)
                    & (active[RAW_URI_FEATURE] < IPFS_PREFIX_LENGTH)
                ).sum()
            ),
        },
        "feature_definition": {
            "name": PAYLOAD_FEATURE,
            "formula": "metadata_uri_length - 7 * metadata_uri_is_ipfs",
            "unit": "characters",
            "availability": (
                "URI and URI scheme are parsed from deployment metadata in the deployment "
                "transaction and are available at t_decision"
            ),
            "semantic_scope": "URI length excluding the fixed ipfs:// scheme prefix",
            "interpretation_limit": (
                "the stored feature does not expose the URI text, so non-IPFS schemes cannot be "
                "normalized and IPFS gateway URLs remain in the non-IPFS group"
            ),
        },
        "active_window_start": active_start.isoformat(),
        "active_window_end": active_end.isoformat(),
        "baseline_feature_names": baseline_features,
        "payload_feature_names": payload_features,
        "feature_replacement": {"removed": RAW_URI_FEATURE, "added": PAYLOAD_FEATURE},
        "hyperparameters": BOOSTING_PARAMETERS,
        "expanding_folds": fold_results,
        "mean_payload_minus_raw_pr_auc": float(np.mean(fold_deltas)),
        "minimum_payload_minus_raw_pr_auc": float(np.min(fold_deltas)),
        "maximum_payload_minus_raw_pr_auc": float(np.max(fold_deltas)),
        "improved_all_development_checks": bool(
            all(delta > 0 for delta in fold_deltas) and standard_delta > 0
        ),
        "standard_validation": {
            "raw_uri_length": standard_raw,
            "uri_payload_length": standard_payload,
            "payload_minus_raw_pr_auc": standard_delta,
            "raw_length_profile": _class_profile(validation, RAW_URI_FEATURE),
            "payload_length_profile": _class_profile(validation, PAYLOAD_FEATURE),
            "raw_to_payload_spearman": float(
                validation[[RAW_URI_FEATURE, PAYLOAD_FEATURE]].corr(method="spearman").iloc[0, 1]
            ),
            "ipfs_fraction_by_class": {
                "not_bought": float(
                    validation.loc[validation["label"] == 0, IPFS_FLAG_FEATURE].mean()
                ),
                "bought": float(validation.loc[validation["label"] == 1, IPFS_FLAG_FEATURE].mean()),
            },
        },
        "max_evaluated_time": max_evaluated_time.isoformat(),
        "final_holdout_start": final_holdout_start.isoformat(),
        "final_holdout_evaluated": False,
        "final_holdout_status": "sealed_no_predictions_generated",
    }


def render_report(metrics: dict[str, object]) -> str:
    decision_text = {
        "supported_improves_all_development_checks": (
            "The payload-length replacement improves PR-AUC in every predeclared development check."
        ),
        "retained_semantically_within_noninferiority_margin": (
            "The payload-length replacement is retained only as a semantically clearer, "
            "noninferior parameterization; it does not improve every check."
        ),
        "rejected_exceeds_noninferiority_margin": (
            "The payload-length replacement is rejected because at least one development check "
            "exceeds the predeclared 0.002 PR-AUC loss margin."
        ),
    }[metrics["decision"]]
    fold_rows = []
    for row in metrics["expanding_folds"]:
        op = row["uri_payload_length"]["selected_operating_point"]
        fold_rows.append(
            "| {fold} | {start} to {end} | {raw:.5f} | {payload:.5f} | {delta:+.5f} | "
            "{precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                fold=row["fold"],
                start=pd.Timestamp(row["uri_payload_length"]["validation_start"]).date(),
                end=pd.Timestamp(row["uri_payload_length"]["validation_end"]).date(),
                raw=row["raw_uri_length"]["population_adjusted_pr_auc"],
                payload=row["uri_payload_length"]["population_adjusted_pr_auc"],
                delta=row["payload_minus_raw_pr_auc"],
                precision=op["precision"],
                recall=op["recall"],
                f1=op["f1"],
            )
        )
    standard = metrics["standard_validation"]
    raw = standard["raw_uri_length"]
    payload = standard["uri_payload_length"]
    raw_op = raw["selected_operating_point"]
    payload_op = payload["selected_operating_point"]
    payload_profiles = standard["payload_length_profile"]
    standard_rows = [
        "| PR-AUC | {raw:.5f} | {payload:.5f} | {delta:+.5f} |".format(
            raw=raw["population_adjusted_pr_auc"],
            payload=payload["population_adjusted_pr_auc"],
            delta=standard["payload_minus_raw_pr_auc"],
        ),
        "| Precision | {raw:.4f} | {payload:.4f} | {delta:+.4f} |".format(
            raw=raw_op["precision"],
            payload=payload_op["precision"],
            delta=payload_op["precision"] - raw_op["precision"],
        ),
        "| Recall | {raw:.4f} | {payload:.4f} | {delta:+.4f} |".format(
            raw=raw_op["recall"],
            payload=payload_op["recall"],
            delta=payload_op["recall"] - raw_op["recall"],
        ),
        "| F1 | {raw:.4f} | {payload:.4f} | {delta:+.4f} |".format(
            raw=raw_op["f1"],
            payload=payload_op["f1"],
            delta=payload_op["f1"] - raw_op["f1"],
        ),
        "| Threshold | {raw:.6f} | {payload:.6f} | n/a |".format(
            raw=raw_op["threshold"], payload=payload_op["threshold"]
        ),
    ]
    return f"""# Metadata URI payload length at deployment

## Decision

{decision_text} This is a two-run-reproduced development result, not an independent final
estimate; the final chronological holdout remains sealed.

| Fold | Validation period | Raw PR-AUC | Payload PR-AUC | Delta | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(fold_rows)}

### Standard validation operating point

| Metric | Raw URI length | URI payload length | Payload minus raw |
|---|---:|---:|---:|
{chr(10).join(standard_rows)}

![Raw URI length versus payload length](figures/metadata_uri_payload.svg)

## Interpretation at `t_decision`

The replacement subtracts the seven-character `ipfs://` scheme prefix from IPFS URIs. URI length
and the IPFS flag are parsed from deployment metadata, so the feature is available at
`t_decision`; no later trade, price, candle, label, realized P&L, or future deployer history enters
the model. The existing IPFS flag remains in both variants, isolating the length reparameterization.

On standard validation, bought deployments have median payload length
{payload_profiles["bought"]["median"]:.1f} characters (p90
{payload_profiles["bought"]["p90"]:.1f}), versus
{payload_profiles["not_bought"]["median"]:.1f} (p90
{payload_profiles["not_bought"]["p90"]:.1f}) for sampled not-bought deployments. The raw-to-payload
Spearman correlation is {standard["raw_to_payload_spearman"]:.6f}. Because URI text was not stored,
other schemes and IPFS gateway URLs cannot be normalized.

## Reproducibility boundary

- Dataset: `{metrics["dataset"]}`; SHA-256 `{metrics["dataset_sha256"]}`.
- Rows: {metrics["schema"]["rows"]:,}; active rows: {metrics["schema"]["active_rows"]:,}; unique
  tokens: {metrics["schema"]["unique_tokens"]:,}; negative payload rows:
  {metrics["schema"]["negative_payload_rows"]}.
- Exactly one feature is replaced on the retained fee-adjusted outflow baseline.
- Two complete deterministic runs matched the metrics dictionary exactly.
- Maximum evaluated time: `{metrics["max_evaluated_time"]}`.
- Final holdout starts: `{metrics["final_holdout_start"]}`; no holdout predictions were generated.
"""


def render_svg(metrics: dict[str, object]) -> str:
    rows = [
        (
            f"Fold {row['fold']}",
            row["raw_uri_length"]["population_adjusted_pr_auc"],
            row["uri_payload_length"]["population_adjusted_pr_auc"],
        )
        for row in metrics["expanding_folds"]
    ]
    standard = metrics["standard_validation"]
    rows.append(
        (
            "Standard",
            standard["raw_uri_length"]["population_adjusted_pr_auc"],
            standard["uri_payload_length"]["population_adjusted_pr_auc"],
        )
    )
    width, height = 860, 330
    left, top, plot_width, plot_height = 90, 80, 700, 190
    maximum = max(max(raw, payload) for _, raw, payload in rows) * 1.12
    group_width = plot_width / len(rows)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Arial" font-size="22" font-weight="700" '
        'fill="#111827">Metadata URI payload length</text>',
        '<text x="28" y="57" font-family="Arial" font-size="13" fill="#4b5563">'
        "Population-weighted PR-AUC; final holdout sealed</text>",
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#9ca3af"/>',
    ]
    for index, (name, raw, payload) in enumerate(rows):
        center = left + group_width * (index + 0.5)
        raw_height = raw / maximum * plot_height
        payload_height = payload / maximum * plot_height
        elements.extend(
            [
                f'<rect x="{center - 35:.1f}" y="{top + plot_height - raw_height:.1f}" '
                f'width="30" height="{raw_height:.1f}" fill="#64748b"/>',
                f'<rect x="{center + 5:.1f}" y="{top + plot_height - payload_height:.1f}" '
                f'width="30" height="{payload_height:.1f}" fill="#0f766e"/>',
                f'<text x="{center:.1f}" y="{top + plot_height + 20}" text-anchor="middle" '
                f'font-family="Arial" font-size="12" fill="#111827">{name}</text>',
                f'<text x="{center - 20:.1f}" y="{top + plot_height - raw_height - 6:.1f}" '
                f'text-anchor="middle" font-family="Arial" font-size="11">{raw:.4f}</text>',
                f'<text x="{center + 20:.1f}" y="{top + plot_height - payload_height - 6:.1f}" '
                f'text-anchor="middle" font-family="Arial" font-size="11">{payload:.4f}</text>',
            ]
        )
    elements.extend(
        [
            '<rect x="570" y="28" width="12" height="12" fill="#64748b"/>',
            '<text x="588" y="39" font-family="Arial" font-size="12">Raw length</text>',
            '<rect x="680" y="28" width="12" height="12" fill="#0f766e"/>',
            '<text x="698" y="39" font-family="Arial" font-size="12">Payload length</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def main() -> None:
    dataset_path = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
    metrics = run_metadata_uri_payload_experiment(dataset_path)
    reproduced = run_metadata_uri_payload_experiment(dataset_path)
    if reproduced != metrics:
        raise AssertionError("deterministic metadata-URI-payload rerun did not match")
    metrics["reproduction_verification"] = {
        "verified": True,
        "run_count": 2,
        "comparison": "complete_metrics_dictionary_exact_match",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_root = REPORT_DIR.parent
    figure_dir = report_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "metadata_uri_payload.json"
    report_path = report_root / "metadata_uri_payload.md"
    figure_path = figure_dir / "metadata_uri_payload.svg"
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
