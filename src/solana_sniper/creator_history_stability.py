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
    positive_times = frame.loc[frame["label"] == 1, "decision_time"]
    frame = frame.loc[
        frame["decision_time"].between(positive_times.min(), positive_times.max())
    ].copy()
    outer = chronological_train_validation_test_split(
        frame,
        time_column="decision_time",
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    pretest = pd.concat([outer.train, outer.validation], ignore_index=True)
    features_with_creator = _numeric_features(pretest)
    features_without_creator = [
        feature for feature in features_with_creator if feature not in CREATOR_HISTORY_FEATURES
    ]
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
    standard_metrics, standard_model = _validation_metrics(
        outer.train, outer.validation, features_with_creator
    )
    attribution = _permutation_attribution(standard_model, outer.validation, features_with_creator)
    max_evaluated_time = max(
        pd.Timestamp(fold["with_creator_history"]["validation_end"]) for fold in fold_results
    )
    test_start = outer.test["decision_time"].min()
    if not max_evaluated_time < test_start:
        raise AssertionError("stability analysis touched the final holdout period")
    deltas = [fold["pr_auc_delta"] for fold in fold_results]
    return {
        "experiment": "creator_history_temporal_stability_and_attribution",
        "single_hypothesis": "creator_history_signal_is_stable_and_interpretable",
        "dataset": project_relative(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "code_parent_commit": git_head(),
        "hyperparameters": BOOSTING_PARAMETERS,
        "creator_features": CREATOR_HISTORY_FEATURES,
        "expanding_folds": fold_results,
        "positive_delta_fold_count": int(sum(delta > 0 for delta in deltas)),
        "fold_count": len(deltas),
        "mean_pr_auc_delta": float(np.mean(deltas)),
        "standard_validation": standard_metrics,
        "permutation_attribution": attribution,
        "max_evaluated_time": max_evaluated_time.isoformat(),
        "final_test_start": test_start.isoformat(),
        "test_status": "withheld_no_predictions_generated",
    }


def main() -> None:
    dataset_path = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
    metrics = run_stability(dataset_path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "creator_history_stability.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    append_experiment({**metrics, "metrics_path": project_relative(output)})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
