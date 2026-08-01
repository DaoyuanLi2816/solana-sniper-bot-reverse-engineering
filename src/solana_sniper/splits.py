from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame


def chronological_split(
    frame: pd.DataFrame,
    *,
    time_column: str,
    validation_fraction: float = 0.2,
) -> TimeSplit:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    ordered = frame.sort_values(time_column, kind="stable").reset_index(drop=True)
    cut = int(len(ordered) * (1 - validation_fraction))
    if cut <= 0 or cut >= len(ordered):
        raise ValueError("split would create an empty partition")
    train = ordered.iloc[:cut].copy()
    validation = ordered.iloc[cut:].copy()
    if train[time_column].max() > validation[time_column].min():
        raise AssertionError("chronological split overlaps in time")
    return TimeSplit(train=train, validation=validation)
