from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame


@dataclass(frozen=True)
class ThreeWayTimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _deterministic_time_order(frame: pd.DataFrame, time_column: str) -> pd.DataFrame:
    order_columns = [time_column]
    for candidate in ("token_address", "tx_hash"):
        if candidate in frame.columns:
            order_columns.append(candidate)
            break
    return frame.sort_values(order_columns, kind="stable").reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame,
    *,
    time_column: str,
    validation_fraction: float = 0.2,
) -> TimeSplit:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    ordered = _deterministic_time_order(frame, time_column)
    cut = int(len(ordered) * (1 - validation_fraction))
    if cut <= 0 or cut >= len(ordered):
        raise ValueError("split would create an empty partition")
    train = ordered.iloc[:cut].copy()
    validation = ordered.iloc[cut:].copy()
    if train[time_column].max() > validation[time_column].min():
        raise AssertionError("chronological split overlaps in time")
    return TimeSplit(train=train, validation=validation)


def chronological_train_validation_test_split(
    frame: pd.DataFrame,
    *,
    time_column: str,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> ThreeWayTimeSplit:
    """Split on timestamp boundaries so the same timestamp never crosses partitions."""
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation_fraction and test_fraction must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction plus test_fraction must be less than one")
    ordered = _deterministic_time_order(frame, time_column)
    unique_times = ordered[time_column].drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_times) < 3:
        raise ValueError("at least three distinct timestamps are required")
    train_fraction = 1 - validation_fraction - test_fraction
    validation_index = max(1, int(len(unique_times) * train_fraction))
    test_index = max(validation_index + 1, int(len(unique_times) * (1 - test_fraction)))
    if test_index >= len(unique_times):
        raise ValueError("split would create an empty partition")
    validation_start = unique_times.iloc[validation_index]
    test_start = unique_times.iloc[test_index]
    train = ordered.loc[ordered[time_column] < validation_start].copy()
    validation = ordered.loc[
        (ordered[time_column] >= validation_start) & (ordered[time_column] < test_start)
    ].copy()
    test = ordered.loc[ordered[time_column] >= test_start].copy()
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("split would create an empty partition")
    if not train[time_column].max() < validation[time_column].min():
        raise AssertionError("train and validation overlap in time")
    if not validation[time_column].max() < test[time_column].min():
        raise AssertionError("validation and test overlap in time")
    return ThreeWayTimeSplit(train=train, validation=validation, test=test)
