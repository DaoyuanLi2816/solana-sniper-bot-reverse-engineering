import pandas as pd
import pytest

from solana_sniper.creator_slot_batching import (
    BATCH_FEATURE,
    add_creator_slot_batching,
    batch_decision,
)


def test_creator_slot_batching_handles_new_and_batching_creators() -> None:
    frame = pd.DataFrame(
        {
            "creator_prior_deploy_count": [0, 2, 10, 4],
            "creator_prior_active_slot_count": [0, 2, 5, 1],
        }
    )
    result = add_creator_slot_batching(frame)
    assert result[BATCH_FEATURE].tolist() == [0.0, 1.0, 2.0, 4.0]
    assert BATCH_FEATURE not in frame.columns


@pytest.mark.parametrize(
    ("counts", "slots", "message"),
    [
        ([1], [2], "cannot exceed"),
        ([1], [0], "both be zero or positive"),
        ([0], [1], "cannot exceed"),
        ([-1], [0], "nonnegative"),
        ([1], [None], "missing"),
    ],
)
def test_creator_slot_batching_rejects_invalid_history(
    counts: list[object], slots: list[object], message: str
) -> None:
    frame = pd.DataFrame(
        {
            "creator_prior_deploy_count": counts,
            "creator_prior_active_slot_count": slots,
        }
    )
    with pytest.raises(ValueError, match=message):
        add_creator_slot_batching(frame)


def test_creator_slot_batching_rejects_missing_column() -> None:
    with pytest.raises(ValueError, match="inputs are missing"):
        add_creator_slot_batching(pd.DataFrame({"creator_prior_deploy_count": [1]}))


def test_batch_decision_requires_every_development_check_to_improve() -> None:
    assert batch_decision([0.001, 0.002, 0.003], 0.001) == (
        "supported_improves_all_development_checks"
    )
    assert batch_decision([0.001, 0.0, 0.003], 0.001) == (
        "rejected_not_consistent_across_development_checks"
    )
    assert batch_decision([0.001, -0.001, 0.003], 0.001) == (
        "rejected_not_consistent_across_development_checks"
    )
    with pytest.raises(ValueError, match="at least one"):
        batch_decision([], 0.0)
