import pandas as pd

from solana_sniper.splits import chronological_split


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
