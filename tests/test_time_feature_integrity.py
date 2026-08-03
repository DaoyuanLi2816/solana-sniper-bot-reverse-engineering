import pandas as pd
import pytest

from solana_sniper.time_feature_integrity import (
    CANONICAL_HOUR,
    CANONICAL_WEEKDAY,
    add_canonical_utc_time,
    integrity_decision,
)


def test_add_canonical_utc_time_uses_hour_and_monday_zero_weekday() -> None:
    frame = pd.DataFrame(
        {
            "decision_time": [
                "2026-01-01T00:00:59Z",
                "2026-03-09T23:15:00Z",
            ]
        }
    )
    result = add_canonical_utc_time(frame)
    assert result[CANONICAL_HOUR].tolist() == [0, 23]
    assert result[CANONICAL_WEEKDAY].tolist() == [3, 0]
    assert CANONICAL_HOUR not in frame.columns


def test_add_canonical_utc_time_rejects_missing_or_invalid_time() -> None:
    with pytest.raises(ValueError, match="is missing"):
        add_canonical_utc_time(pd.DataFrame({"other": [1]}))
    with pytest.raises(ValueError):
        add_canonical_utc_time(pd.DataFrame({"decision_time": ["not-a-time"]}))
    with pytest.raises(ValueError, match="contains missing"):
        add_canonical_utc_time(pd.DataFrame({"decision_time": [None]}))


def test_integrity_decision_requires_fix_for_any_mismatch() -> None:
    assert integrity_decision(0, 0) == "already_consistent"
    assert integrity_decision(1, 0) == "required_integrity_fix"
    assert integrity_decision(0, 1) == "required_integrity_fix"
    with pytest.raises(ValueError, match="nonnegative"):
        integrity_decision(-1, 0)
