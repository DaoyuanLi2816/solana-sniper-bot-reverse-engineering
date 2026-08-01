"""Nonlinear comparison using the same leak-free features and chronological split."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from solana_sniper.baseline import (
    NEGATIVE_SAMPLE_WEIGHT,
    NON_FEATURE_COLUMNS,
    _best_f1_threshold,
    _fixed_threshold_metrics,
    _population_weights,
)
from solana_sniper.guardrails import assert_feature_names_are_pre_decision
from solana_sniper.manifest import append_experiment, sha256_file
from solana_sniper.paths import PROCESSED_DIR, REPORT_DIR, project_relative
from solana_sniper.splits import chronological_train_validation_test_split


def run_boosting(
    dataset_path: Path,
    *,
    experiment: str = "hist_gradient_boosting_predecision",
    single_change: str = "model_family_logistic_to_hist_gradient_boosting",
    evaluate_test: bool = True,
) -> dict[str, object]:
    frame = pd.read_parquet(dataset_path)
    positive_times = frame.loc[frame["label"] == 1, "decision_time"]
    active_start = positive_times.min()
    active_end = positive_times.max()
    frame = frame.loc[frame["decision_time"].between(active_start, active_end)].copy()
    numeric_features = [
        column
        for column in frame.select_dtypes(include=["number", "bool"]).columns
        if column not in NON_FEATURE_COLUMNS
    ]
    assert_feature_names_are_pre_decision(numeric_features)
    split = chronological_train_validation_test_split(
        frame,
        time_column="decision_time",
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=0.1,
        class_weight="balanced",
        random_state=20260801,
    )
    model.fit(split.train[numeric_features], split.train["label"])
    validation_probabilities = model.predict_proba(split.validation[numeric_features])[:, 1]
    validation_weights = _population_weights(split.validation["label"])
    operating_point = _best_f1_threshold(
        split.validation["label"], validation_probabilities, validation_weights
    )
    result: dict[str, object] = {
        "experiment": experiment,
        "single_change_from_baseline": single_change,
        "dataset": project_relative(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "active_window_start": active_start.isoformat(),
        "active_window_end": active_end.isoformat(),
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "test_rows": int(len(split.test)),
        "negative_sample_weight": NEGATIVE_SAMPLE_WEIGHT,
        "validation_population_adjusted_pr_auc": float(
            average_precision_score(
                split.validation["label"],
                validation_probabilities,
                sample_weight=validation_weights,
            )
        ),
        "validation_population_adjusted_prevalence": float(
            np.average(split.validation["label"].to_numpy(), weights=validation_weights)
        ),
        "validation_selected_population_operating_point": operating_point,
        "test_status": "evaluated" if evaluate_test else "withheld_until_candidate_freeze",
        "feature_names": numeric_features,
        "hyperparameters": model.get_params(),
    }
    if evaluate_test:
        test_probabilities = model.predict_proba(split.test[numeric_features])[:, 1]
        test_weights = _population_weights(split.test["label"])
        result["test_metrics_at_validation_threshold"] = _fixed_threshold_metrics(
            split.test["label"], test_probabilities, operating_point["threshold"], test_weights
        )
    return result


def main() -> None:
    dataset_path = PROCESSED_DIR / "classification_dataset.parquet"
    metrics = run_boosting(dataset_path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "boosting_metrics.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    append_experiment({**metrics, "metrics_path": project_relative(output)})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
