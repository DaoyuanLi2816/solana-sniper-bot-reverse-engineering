import pandas as pd

from solana_sniper.splits import (
    chronological_split,
    chronological_train_validation_test_split,
)


def test_chronological_split_is_strictly_ordered() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-03", "2026-01-01", "2026-01-04", "2026-01-02"], utc=True
            ),
            "value": [3, 1, 4, 2],
        }
    )
    split = chronological_split(frame, time_column="time", validation_fraction=0.5)
    assert split.train["value"].tolist() == [1, 2]
    assert split.validation["value"].tolist() == [3, 4]


def test_three_way_split_keeps_tied_timestamps_together() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                ],
                utc=True,
            ),
            "value": [1, 2, 3, 4, 5, 6],
        }
    )
    split = chronological_train_validation_test_split(
        frame,
        time_column="time",
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    assert split.train["time"].max() < split.validation["time"].min()
    assert split.validation["time"].max() < split.test["time"].min()
    partitions = [split.train, split.validation, split.test]
    assert (
        sum((part["time"] == pd.Timestamp("2026-01-02", tz="UTC")).sum() for part in partitions)
        == 2
    )
    assert (
        sum(
            bool((part["time"] == pd.Timestamp("2026-01-02", tz="UTC")).any())
            for part in partitions
        )
        == 1
    )
