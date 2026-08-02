import pandas as pd
import pytest

from solana_sniper.priority_fee_intensity import (
    INTENSITY_FEATURE,
    NONINFERIORITY_MARGIN,
    add_priority_fee_intensity,
    intensity_decision,
)


def test_priority_intensity_divides_fee_by_positive_compute_units() -> None:
    frame = pd.DataFrame(
        {
            "priority_fee_lamports": [0, 100, 75],
            "compute_units": [10, 20, 25],
        }
    )
    result = add_priority_fee_intensity(frame)
    assert result[INTENSITY_FEATURE].tolist() == [0, 5, 3]
    assert INTENSITY_FEATURE not in frame.columns


def test_priority_intensity_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="missing"):
        add_priority_fee_intensity(pd.DataFrame({"priority_fee_lamports": [1]}))
    with pytest.raises(ValueError, match="missing values"):
        add_priority_fee_intensity(
            pd.DataFrame(
                {
                    "priority_fee_lamports": [1, None],
                    "compute_units": [2, 2],
                }
            )
        )
    with pytest.raises(ValueError, match="strictly positive"):
        add_priority_fee_intensity(
            pd.DataFrame(
                {
                    "priority_fee_lamports": [1, 1],
                    "compute_units": [2, 0],
                }
            )
        )


def test_intensity_decision_separates_improvement_noninferiority_and_rejection() -> None:
    assert intensity_decision([0.001, 0.002, 0.003], 0.001) == (
        "supported_improves_all_development_checks"
    )
    assert intensity_decision([0.001, -NONINFERIORITY_MARGIN, 0.0], 0.001) == (
        "retained_semantically_within_noninferiority_margin"
    )
    assert intensity_decision([0.001, -NONINFERIORITY_MARGIN - 1e-6, 0.0], 0.001) == (
        "rejected_exceeds_noninferiority_margin"
    )
    with pytest.raises(ValueError, match="at least one"):
        intensity_decision([], 0.0)
