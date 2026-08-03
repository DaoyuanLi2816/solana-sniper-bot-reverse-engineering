import pandas as pd
import pytest

from solana_sniper.creator_history_stability import (
    expanding_time_folds,
    stability_decision,
)


def test_expanding_folds_are_strict_and_cover_only_later_times() -> None:
    frame = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-09",
                    "2026-01-10",
                ],
                utc=True,
            ),
            "value": range(11),
        }
    )
    folds = expanding_time_folds(frame, time_column="decision_time")
    assert len(folds) == 3
    for fold in folds:
        assert fold.train["decision_time"].max() < fold.validation["decision_time"].min()
    assert len(folds[0].train) < len(folds[1].train) < len(folds[2].train)
    assert folds[-1].validation["decision_time"].max() == frame["decision_time"].max()


def test_stability_decision_requires_every_development_check_to_improve() -> None:
    assert stability_decision([0.01, 0.02, 0.001], 0.003) == (
        "supported_improves_all_development_checks"
    )
    assert stability_decision([0.01, -0.001, 0.002], 0.003) == (
        "rejected_not_consistent_across_development_checks"
    )
    assert stability_decision([0.01, 0.002, 0.003], 0.0) == (
        "rejected_not_consistent_across_development_checks"
    )
    with pytest.raises(ValueError, match="at least one"):
        stability_decision([], 0.001)
