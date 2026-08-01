import pandas as pd

from solana_sniper.creator_history_stability import expanding_time_folds


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
