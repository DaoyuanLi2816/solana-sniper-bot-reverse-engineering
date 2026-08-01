import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from solana_sniper.guardrails import assert_feature_names_are_pre_decision
from solana_sniper.manifest import append_experiment, sha256_file
from solana_sniper.paths import PROCESSED_DIR, REPORT_DIR
from solana_sniper.splits import chronological_split

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


def run_baseline(dataset_path: Path) -> dict[str, object]:
    frame = pd.read_parquet(dataset_path)
    if frame["label"].nunique() != 2:
        raise ValueError("Classification baseline requires both bought and not-bought examples")
    numeric_features = [
        column
        for column in frame.select_dtypes(include=["number", "bool"]).columns
        if column not in NON_FEATURE_COLUMNS
    ]
    assert_feature_names_are_pre_decision(numeric_features)
    split = chronological_split(frame, time_column="decision_time", validation_fraction=0.2)
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
    probabilities = model.predict_proba(split.validation)[:, 1]
    metrics = {
        "experiment": "logistic_predecision_baseline",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "validation_positive_rate": float(split.validation["label"].mean()),
        "validation_pr_auc": float(
            average_precision_score(split.validation["label"], probabilities)
        ),
        "best_f1_operating_point": _best_f1_threshold(split.validation["label"], probabilities),
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
