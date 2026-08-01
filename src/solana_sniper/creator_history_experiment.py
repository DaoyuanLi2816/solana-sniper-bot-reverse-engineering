"""Evaluate creator-history features on validation while preserving the final holdout."""

import json

from solana_sniper.boosting import run_boosting
from solana_sniper.manifest import append_experiment, git_head, sha256_file
from solana_sniper.paths import PROCESSED_DIR, REPORT_DIR, project_relative


def main() -> None:
    dataset_path = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
    metrics = run_boosting(
        dataset_path,
        experiment="hist_gradient_boosting_creator_history",
        single_change="add_strict_prior_creator_deploy_frequency_and_recency",
        evaluate_test=False,
    )
    metrics["code_parent_commit"] = git_head()
    metrics["artifact_sha256"] = sha256_file(dataset_path)
    metrics["decision"] = "compare_validation_pr_auc_only_final_test_withheld"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "creator_history_metrics.json"
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    append_experiment({**metrics, "metrics_path": project_relative(output)})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
