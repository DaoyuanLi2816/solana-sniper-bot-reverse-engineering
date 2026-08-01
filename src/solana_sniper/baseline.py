import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from solana_sniper.guardrails import assert_feature_names_are_pre_decision
from solana_sniper.manifest import append_experiment, sha256_file
from solana_sniper.paths import PROCESSED_DIR, REPORT_DIR
from solana_sniper.splits import chronological_train_validation_test_split

NON_FEATURE_COLUMNS = {
    "label",
    "decision_time",
    "tx_hash",
    "token_address",
    "tx_signer",
    "creator_address",
    "line_number",
    "blockTime",
    "blockSlot",
    "block_slot",
}


def _best_f1_threshold(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.nanargmax(f1[:-1]))
    return {
        "threshold": float(thresholds[index]),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
    }


def _fixed_threshold_metrics(
    labels: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "prevalence_baseline_pr_auc": float(labels.mean()),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run_baseline(dataset_path: Path) -> dict[str, object]:
    frame = pd.read_parquet(dataset_path)
    if frame["label"].nunique() != 2:
        raise ValueError("Classification baseline requires both bought and not-bought examples")
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
    for name, partition in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        if partition["label"].nunique() != 2:
            raise ValueError(f"{name} partition does not contain both classes")
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]),
                numeric_features,
            )
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "model",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=20260801),
            ),
        ]
    )
    model.fit(split.train, split.train["label"])
    validation_probabilities = model.predict_proba(split.validation)[:, 1]
    operating_point = _best_f1_threshold(split.validation["label"], validation_probabilities)
    test_probabilities = model.predict_proba(split.test)[:, 1]
    metrics = {
        "experiment": "logistic_predecision_baseline",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "active_window_start": active_start.isoformat(),
        "active_window_end": active_end.isoformat(),
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "test_rows": int(len(split.test)),
        "train_end": split.train["decision_time"].max().isoformat(),
        "validation_end": split.validation["decision_time"].max().isoformat(),
        "validation_positive_rate": float(split.validation["label"].mean()),
        "validation_pr_auc": float(
            average_precision_score(split.validation["label"], validation_probabilities)
        ),
        "validation_selected_operating_point": operating_point,
        "test_metrics_at_validation_threshold": _fixed_threshold_metrics(
            split.test["label"], test_probabilities, operating_point["threshold"]
        ),
        "feature_names": numeric_features,
    }
    return metrics


def main() -> None:
    dataset_path = PROCESSED_DIR / "classification_dataset.parquet"
    metrics = run_baseline(dataset_path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "baseline_metrics.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    append_experiment({**metrics, "metrics_path": str(output)})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
