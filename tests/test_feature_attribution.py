import numpy as np
import pandas as pd
import pytest

from solana_sniper.feature_attribution import (
    FEATURE_GROUPS,
    aggregate_importance,
    permutation_attribution,
    validate_feature_groups,
)


class _SignalModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probability = np.clip(frame["creator_prior_deploy_count"].to_numpy() / 4, 0.01, 0.99)
        return np.column_stack([1 - probability, probability])


def _feature_frame() -> pd.DataFrame:
    rows = 12
    data = {feature: np.zeros(rows) for columns in FEATURE_GROUPS.values() for feature in columns}
    data["creator_prior_deploy_count"] = np.array([0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4])
    data["label"] = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    return pd.DataFrame(data)


def test_feature_groups_are_disjoint_and_complete() -> None:
    features = [feature for columns in FEATURE_GROUPS.values() for feature in columns]
    validate_feature_groups(features)
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_feature_groups(features[:-1])


def test_permutation_attribution_is_deterministic_and_finds_signal() -> None:
    frame = _feature_frame()
    features = [feature for columns in FEATURE_GROUPS.values() for feature in columns]
    first = permutation_attribution(_SignalModel(), frame, features, repeats=3, seed_offset=1)
    second = permutation_attribution(_SignalModel(), frame, features, repeats=3, seed_offset=1)
    assert first == second
    assert first["groups"]["creator_history"]["pr_auc_drop_mean"] > 0
    assert first["features"]["creator_prior_deploy_count"]["pr_auc_drop_mean"] > 0


def test_aggregate_importance_ranks_mean_temporal_drop() -> None:
    folds = [
        {
            "attribution": {
                "groups": {
                    "creator_history": {"pr_auc_drop_mean": 0.3},
                    "metadata": {"pr_auc_drop_mean": 0.1},
                }
            }
        },
        {
            "attribution": {
                "groups": {
                    "creator_history": {"pr_auc_drop_mean": 0.2},
                    "metadata": {"pr_auc_drop_mean": -0.1},
                }
            }
        },
    ]
    standard = {
        "groups": {
            "creator_history": {"pr_auc_drop_mean": 0.25},
            "metadata": {"pr_auc_drop_mean": 0.0},
        }
    }
    ranked = aggregate_importance(folds, standard, result_key="groups")
    assert ranked[0]["name"] == "creator_history"
    assert ranked[0]["positive_fold_count"] == 2
    assert ranked[1]["positive_fold_count"] == 1
