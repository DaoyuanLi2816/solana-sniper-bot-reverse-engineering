import pandas as pd
import pytest

from solana_sniper.creator_deployment_rate import (
    RATE_FEATURE,
    add_creator_deployment_rate,
    rate_decision,
)


def test_creator_deployment_rate_handles_new_and_observed_creators() -> None:
    frame = pd.DataFrame(
        {
            "creator_prior_deploy_count": [0, 2, 10, 4],
            "creator_observed_age_seconds": [None, 0, 5 * 86_400, 2 * 86_400],
        }
    )
    result = add_creator_deployment_rate(frame)
    assert result[RATE_FEATURE].tolist() == [0.0, 2.0, 2.0, 2.0]
    assert RATE_FEATURE not in frame.columns


def test_creator_deployment_rate_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="missing"):
        add_creator_deployment_rate(pd.DataFrame({"creator_prior_deploy_count": [1]}))
    with pytest.raises(ValueError, match="nonnegative"):
        add_creator_deployment_rate(
            pd.DataFrame(
                {
                    "creator_prior_deploy_count": [-1],
                    "creator_observed_age_seconds": [1],
                }
            )
        )
    with pytest.raises(ValueError, match="only when prior count is zero"):
        add_creator_deployment_rate(
            pd.DataFrame(
                {
                    "creator_prior_deploy_count": [1],
                    "creator_observed_age_seconds": [None],
                }
            )
        )


def test_rate_decision_requires_every_development_check_to_improve() -> None:
    assert rate_decision([0.001, 0.002, 0.003], 0.001) == (
        "supported_improves_all_development_checks"
    )
    assert rate_decision([0.001, 0.0, 0.003], 0.001) == (
        "rejected_not_consistent_across_development_checks"
    )
    assert rate_decision([0.001, -0.001, 0.003], 0.001) == (
        "rejected_not_consistent_across_development_checks"
    )
    with pytest.raises(ValueError, match="at least one"):
        rate_decision([], 0.0)
