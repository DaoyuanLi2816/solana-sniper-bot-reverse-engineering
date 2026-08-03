"""Test an interpretable deployment-signer outflow proxy on development time folds."""

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
from solana_sniper.guardrails import assert_feature_names_are_pre_decision
from solana_sniper.manifest import append_experiment, git_head, sha256_file
from solana_sniper.paths import PROCESSED_DIR, REPORT_DIR, project_relative
from solana_sniper.splits import chronological_train_validation_test_split

RAW_FEATURE = "signer_lamport_delta"
PROXY_FEATURE = "deployment_signer_nonfee_outflow_lamports"
NONINFERIORITY_MARGIN = 0.002


def add_deployment_outflow_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    """Add max(-signer balance delta - transaction fee, 0) without mutating input."""
    required = {RAW_FEATURE, "tx_fee_lamports"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"outflow proxy inputs are missing: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError("outflow proxy inputs contain missing values")
    result = frame.copy()
    delta = pd.to_numeric(result[RAW_FEATURE], errors="raise")
    fee = pd.to_numeric(result["tx_fee_lamports"], errors="raise")
    result[PROXY_FEATURE] = (-delta - fee).clip(lower=0)
    if (result[PROXY_FEATURE] < 0).any():
        raise AssertionError("outflow proxy must be nonnegative")
    return result


def noninferiority_decision(deltas: list[float], standard_delta: float) -> str:
    if not deltas:
        raise ValueError("at least one temporal delta is required")
    accepted = min([*deltas, standard_delta]) >= -NONINFERIORITY_MARGIN
    return (
        "supported_within_predeclared_noninferiority_margin"
        if accepted
        else "rejected_proxy_exceeds_noninferiority_margin"
    )


def _performance_interpretation(*, improved_all: bool, accepted: bool) -> str:
    if improved_all:
        return "It increased PR-AUC in all three expanding folds and standard validation."
    if accepted:
        return (
            "It did not increase PR-AUC in every development check, so retention is a semantic "
            "noninferiority decision rather than a performance-improvement claim."
        )
    return "It is therefore not retained as the current selector feature."


def _class_profile(frame: pd.DataFrame, column: str) -> dict[str, object]:
    profiles = {}
    for label, name in ((0, "not_bought"), (1, "bought")):
        values = frame.loc[frame["label"] == label, column].astype(float) / 1_000_000_000
        profiles[name] = {
            "rows": int(len(values)),
            "p10_sol": float(values.quantile(0.1)),
            "median_sol": float(values.median()),
            "p90_sol": float(values.quantile(0.9)),
            "zero_fraction": float((values == 0).mean()),
        }
    return profiles


def _identity_audit(frame: pd.DataFrame) -> dict[str, object]:
    creator_present = frame["creator_address"].notna()
    equals = (
        frame.loc[creator_present, "tx_signer"] == frame.loc[creator_present, "creator_address"]
    )
    return {
        "rows": int(len(frame)),
        "tx_signer_nonnull_fraction": float(frame["tx_signer"].notna().mean()),
        "creator_address_nonnull_rows": int(creator_present.sum()),
        "creator_address_nonnull_fraction": float(creator_present.mean()),
        "signer_equals_creator_among_creator_nonnull_fraction": (
            float(equals.mean()) if len(equals) else None
        ),
        "positive_creator_address_nonnull_rows": int(
            frame.loc[frame["label"] == 1, "creator_address"].notna().sum()
        ),
        "negative_creator_address_nonnull_rows": int(
            frame.loc[frame["label"] == 0, "creator_address"].notna().sum()
        ),
    }


def run_outflow_proxy_experiment(dataset_path: Path) -> dict[str, object]:
    frame = pd.read_parquet(dataset_path)
    required = {
        "label",
        "decision_time",
        "token_address",
        "tx_hash",
        "tx_signer",
        "creator_address",
        RAW_FEATURE,
        "tx_fee_lamports",
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
    split = chronological_train_validation_test_split(
        active,
        time_column="decision_time",
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    pretest = pd.concat([split.train, split.validation], ignore_index=True)
    all_numeric = _numeric_features(pretest)
    raw_features = [feature for feature in all_numeric if feature != PROXY_FEATURE]
    if RAW_FEATURE not in raw_features:
        raise ValueError(f"raw feature {RAW_FEATURE} is missing")
    proxy_features = [
        PROXY_FEATURE if feature == RAW_FEATURE else feature for feature in raw_features
    ]
    assert_feature_names_are_pre_decision(raw_features)
    assert_feature_names_are_pre_decision(proxy_features)
    if len(raw_features) != len(proxy_features) or len(set(proxy_features)) != len(proxy_features):
        raise AssertionError("single-feature replacement changed feature cardinality")

    fold_results = []
    for fold_number, fold in enumerate(
        expanding_time_folds(pretest, time_column="decision_time"), start=1
    ):
        raw_metrics, _ = _validation_metrics(fold.train, fold.validation, raw_features)
        proxy_metrics, _ = _validation_metrics(fold.train, fold.validation, proxy_features)
        delta = float(
            proxy_metrics["population_adjusted_pr_auc"] - raw_metrics["population_adjusted_pr_auc"]
        )
        fold_results.append(
            {
                "fold": fold_number,
                "raw_signer_delta": raw_metrics,
                "fee_adjusted_outflow_proxy": proxy_metrics,
                "proxy_minus_raw_pr_auc": delta,
            }
        )

    standard_raw, _ = _validation_metrics(split.train, split.validation, raw_features)
    standard_proxy, _ = _validation_metrics(split.train, split.validation, proxy_features)
    standard_delta = float(
        standard_proxy["population_adjusted_pr_auc"] - standard_raw["population_adjusted_pr_auc"]
    )
    fold_deltas = [float(row["proxy_minus_raw_pr_auc"]) for row in fold_results]

    max_evaluated_time = max(
        split.validation["decision_time"].max(),
        max(
            pd.Timestamp(row["fee_adjusted_outflow_proxy"]["validation_end"])
            for row in fold_results
        ),
    )
    final_holdout_start = split.test["decision_time"].min()
    if not max_evaluated_time < final_holdout_start:
        raise AssertionError("outflow proxy experiment touched the final holdout")

    return {
        "experiment": "deployment_signer_fee_adjusted_outflow_proxy",
        "single_hypothesis": (
            "replacing_raw_signer_balance_delta_with_fee_adjusted_nonfee_outflow_preserves_"
            "chronological_classification_quality_and_improves_semantic_precision"
        ),
        "decision": noninferiority_decision(fold_deltas, standard_delta),
        "predeclared_noninferiority_margin_pr_auc": NONINFERIORITY_MARGIN,
        "dataset": project_relative(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "code_parent_commit": git_head(),
        "schema": {
            "rows": int(len(frame)),
            "active_rows": int(len(active)),
            "columns": int(len(frame.columns)),
            "unique_tokens": int(frame["token_address"].nunique()),
            "duplicate_tx_hash_rows": int(frame["tx_hash"].duplicated().sum()),
            "positive_rows": int((frame["label"] == 1).sum()),
            "negative_rows": int((frame["label"] == 0).sum()),
        },
        "proxy_definition": {
            "name": PROXY_FEATURE,
            "formula": "max(-signer_lamport_delta - tx_fee_lamports, 0)",
            "unit": "lamports",
            "availability": (
                "deployment transaction meta balances and fee, available when the deployment "
                "transaction is observed at t_decision"
            ),
            "semantic_scope": "deployment signer or fee payer net nonfee lamport outflow",
            "not_proven": (
                "pure creator dev-buy; value can mix account rent, account funding, protocol "
                "transfers, and purchase principal"
            ),
        },
        "identity_audit": _identity_audit(frame),
        "active_window_start": active_start.isoformat(),
        "active_window_end": active_end.isoformat(),
        "raw_feature_names": raw_features,
        "proxy_feature_names": proxy_features,
        "hyperparameters": BOOSTING_PARAMETERS,
        "expanding_folds": fold_results,
        "mean_proxy_minus_raw_pr_auc": float(np.mean(fold_deltas)),
        "minimum_proxy_minus_raw_pr_auc": float(np.min(fold_deltas)),
        "maximum_proxy_minus_raw_pr_auc": float(np.max(fold_deltas)),
        "proxy_pr_auc_improved_all_development_checks": bool(
            all(delta > 0 for delta in fold_deltas) and standard_delta > 0
        ),
        "standard_validation": {
            "raw_signer_delta": standard_raw,
            "fee_adjusted_outflow_proxy": standard_proxy,
            "proxy_minus_raw_pr_auc": standard_delta,
            "outflow_profile": _class_profile(split.validation, PROXY_FEATURE),
            "raw_to_proxy_spearman": float(
                split.validation[[RAW_FEATURE, PROXY_FEATURE]].corr(method="spearman").iloc[0, 1]
            ),
        },
        "max_evaluated_time": max_evaluated_time.isoformat(),
        "final_holdout_start": final_holdout_start.isoformat(),
        "final_holdout_evaluated": False,
        "final_holdout_status": "sealed_no_predictions_generated",
    }


def render_report(metrics: dict[str, object]) -> str:
    accepted = metrics["decision"] == "supported_within_predeclared_noninferiority_margin"
    decision = (
        "The fee-adjusted deployment-signer outflow proxy is retained for interpretation. "
        "It stayed within the predeclared 0.002 PR-AUC noninferiority margin in every "
        "development check."
        if accepted
        else "The proxy is rejected because at least one development check exceeded the "
        "predeclared 0.002 PR-AUC noninferiority margin."
    )
    performance_interpretation = _performance_interpretation(
        improved_all=metrics["proxy_pr_auc_improved_all_development_checks"],
        accepted=accepted,
    )
    fold_rows = []
    for row in metrics["expanding_folds"]:
        proxy_op = row["fee_adjusted_outflow_proxy"]["selected_operating_point"]
        fold_rows.append(
            "| {fold} | {start} to {end} | {raw:.5f} | {proxy:.5f} | {delta:+.5f} | "
            "{precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                fold=row["fold"],
                start=pd.Timestamp(row["fee_adjusted_outflow_proxy"]["validation_start"]).date(),
                end=pd.Timestamp(row["fee_adjusted_outflow_proxy"]["validation_end"]).date(),
                raw=row["raw_signer_delta"]["population_adjusted_pr_auc"],
                proxy=row["fee_adjusted_outflow_proxy"]["population_adjusted_pr_auc"],
                delta=row["proxy_minus_raw_pr_auc"],
                precision=proxy_op["precision"],
                recall=proxy_op["recall"],
                f1=proxy_op["f1"],
            )
        )
    standard = metrics["standard_validation"]
    raw_standard = standard["raw_signer_delta"]
    proxy_standard = standard["fee_adjusted_outflow_proxy"]
    raw_op = raw_standard["selected_operating_point"]
    standard_op = proxy_standard["selected_operating_point"]
    standard_rows = [
        (
            "| PR-AUC | {raw:.5f} | {proxy:.5f} | {delta:+.5f} |".format(
                raw=raw_standard["population_adjusted_pr_auc"],
                proxy=proxy_standard["population_adjusted_pr_auc"],
                delta=standard["proxy_minus_raw_pr_auc"],
            )
        ),
        "| Precision | {raw:.4f} | {proxy:.4f} | {delta:+.4f} |".format(
            raw=raw_op["precision"],
            proxy=standard_op["precision"],
            delta=standard_op["precision"] - raw_op["precision"],
        ),
        "| Recall | {raw:.4f} | {proxy:.4f} | {delta:+.4f} |".format(
            raw=raw_op["recall"],
            proxy=standard_op["recall"],
            delta=standard_op["recall"] - raw_op["recall"],
        ),
        "| F1 | {raw:.4f} | {proxy:.4f} | {delta:+.4f} |".format(
            raw=raw_op["f1"],
            proxy=standard_op["f1"],
            delta=standard_op["f1"] - raw_op["f1"],
        ),
        (
            "| Selected threshold | {raw:.6f} | {proxy:.6f} | n/a |".format(
                raw=raw_op["threshold"], proxy=standard_op["threshold"]
            )
        ),
    ]
    bought = standard["outflow_profile"]["bought"]
    not_bought = standard["outflow_profile"]["not_bought"]
    identity = metrics["identity_audit"]
    return f"""# Deployment-signer fee-adjusted outflow proxy

## Decision

{decision} {performance_interpretation} The full metrics dictionary matched exactly across two
deterministic runs. These are development-period comparisons, not an independent final estimate;
the final chronological holdout remains sealed.

| Fold | Validation period | Raw PR-AUC | Proxy PR-AUC | Delta | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(fold_rows)}

### Standard validation operating point

| Metric | Raw balance delta | Fee-adjusted outflow | Proxy minus raw |
|---|---:|---:|---:|
{chr(10).join(standard_rows)}

![Raw balance delta versus fee-adjusted outflow](figures/deployment_outflow_proxy.svg)

## Exact meaning at `t_decision`

The proxy is `max(-signer_lamport_delta - tx_fee_lamports, 0)`. Both inputs come from the
deployment transaction's balance and fee metadata, so no later trade, candle, outcome, or future
history enters the feature.

The precise name is **deployment-signer/fee-payer net nonfee outflow**, not creator dev-buy.
`creator_address` is present in only {identity["creator_address_nonnull_fraction"]:.2%} of rows
({identity["creator_address_nonnull_rows"]:,}/{identity["rows"]:,}); among those rows, its equality
rate with `tx_signer` is {identity["signer_equals_creator_among_creator_nonnull_fraction"]:.2%}.
Without decoding every program transfer, the net outflow may mix buy principal, account rent,
account funding, and protocol transfers.

## Interpretable association

On standard validation, bought tokens have median signer nonfee outflow of
{bought["median_sol"]:.3f} SOL (p10 {bought["p10_sol"]:.3f}, p90 {bought["p90_sol"]:.3f}), versus
{not_bought["median_sol"]:.3f} SOL for sampled not-bought tokens (p10
{not_bought["p10_sol"]:.3f}, p90 {not_bought["p90_sol"]:.3f}). The raw-to-proxy Spearman
correlation is {standard["raw_to_proxy_spearman"]:.6f}.

The defensible rule hypothesis is therefore: **the target favors deployment transactions whose
signer commits substantially more nonfee lamports**, alongside the previously established signer
history signal. Calling this amount a pure dev-buy would exceed the evidence.

## Reproducibility and boundary

- Dataset: `{metrics["dataset"]}`; SHA-256 `{metrics["dataset_sha256"]}`.
- Rows: {metrics["schema"]["rows"]:,}; unique tokens: {metrics["schema"]["unique_tokens"]:,};
  duplicate transaction-hash rows: {metrics["schema"]["duplicate_tx_hash_rows"]:,}.
- Model: the frozen HGB parameters, with exactly one feature replaced and feature count unchanged.
- Reproduction: {metrics["reproduction_verification"]["run_count"]} deterministic runs matched
  the complete metrics dictionary exactly.
- Maximum evaluated time: `{metrics["max_evaluated_time"]}`.
- Final holdout starts: `{metrics["final_holdout_start"]}`; no holdout predictions were generated.
- Code parent: `{metrics["code_parent_commit"]}`.
"""


def render_svg(metrics: dict[str, object]) -> str:
    rows = [
        (
            f"Fold {row['fold']}",
            row["raw_signer_delta"]["population_adjusted_pr_auc"],
            row["fee_adjusted_outflow_proxy"]["population_adjusted_pr_auc"],
        )
        for row in metrics["expanding_folds"]
    ]
    standard = metrics["standard_validation"]
    rows.append(
        (
            "Standard",
            standard["raw_signer_delta"]["population_adjusted_pr_auc"],
            standard["fee_adjusted_outflow_proxy"]["population_adjusted_pr_auc"],
        )
    )
    width, height = 860, 330
    left, top, plot_width, plot_height = 90, 80, 700, 190
    maximum = max(max(raw, proxy) for _, raw, proxy in rows) * 1.12
    group_width = plot_width / len(rows)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="Arial" font-size="22" font-weight="700" '
        'fill="#111827">Fee-adjusted deployment-signer outflow</text>',
        '<text x="28" y="57" font-family="Arial" font-size="13" fill="#4b5563">'
        "Population-weighted PR-AUC; final holdout sealed</text>",
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#9ca3af"/>',
    ]
    for index, (name, raw, proxy) in enumerate(rows):
        center = left + group_width * (index + 0.5)
        raw_height = raw / maximum * plot_height
        proxy_height = proxy / maximum * plot_height
        elements.extend(
            [
                f'<rect x="{center - 35:.1f}" y="{top + plot_height - raw_height:.1f}" '
                f'width="30" height="{raw_height:.1f}" fill="#64748b"/>',
                f'<rect x="{center + 5:.1f}" y="{top + plot_height - proxy_height:.1f}" '
                f'width="30" height="{proxy_height:.1f}" fill="#0f766e"/>',
                f'<text x="{center:.1f}" y="{top + plot_height + 20}" text-anchor="middle" '
                f'font-family="Arial" font-size="12" fill="#111827">{name}</text>',
                f'<text x="{center - 20:.1f}" y="{top + plot_height - raw_height - 6:.1f}" '
                f'text-anchor="middle" font-family="Arial" font-size="11">{raw:.4f}</text>',
                f'<text x="{center + 20:.1f}" y="{top + plot_height - proxy_height - 6:.1f}" '
                f'text-anchor="middle" font-family="Arial" font-size="11">{proxy:.4f}</text>',
            ]
        )
    elements.extend(
        [
            '<rect x="570" y="28" width="12" height="12" fill="#64748b"/>',
            '<text x="588" y="39" font-family="Arial" font-size="12">Raw delta</text>',
            '<rect x="670" y="28" width="12" height="12" fill="#0f766e"/>',
            '<text x="688" y="39" font-family="Arial" font-size="12">Nonfee outflow</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def main() -> None:
    dataset_path = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
    metrics = run_outflow_proxy_experiment(dataset_path)
    reproduced = run_outflow_proxy_experiment(dataset_path)
    if reproduced != metrics:
        raise AssertionError("deterministic outflow-proxy rerun did not match")
    metrics["reproduction_verification"] = {
        "verified": True,
        "run_count": 2,
        "comparison": "complete_metrics_dictionary_exact_match",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_root = REPORT_DIR.parent
    figure_dir = report_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "deployment_outflow_proxy.json"
    report_path = report_root / "deployment_outflow_proxy.md"
    figure_path = figure_dir / "deployment_outflow_proxy.svg"
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
