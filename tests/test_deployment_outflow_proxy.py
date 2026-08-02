import pandas as pd
import pytest

from solana_sniper.deployment_outflow_proxy import (
    NONINFERIORITY_MARGIN,
    PROXY_FEATURE,
    add_deployment_outflow_proxy,
    noninferiority_decision,
)


def test_outflow_proxy_subtracts_fee_and_clips_at_zero() -> None:
    frame = pd.DataFrame(
        {
            "signer_lamport_delta": [-100, -3, 20],
            "tx_fee_lamports": [5, 5, 5],
        }
    )
    result = add_deployment_outflow_proxy(frame)
    assert result[PROXY_FEATURE].tolist() == [95, 0, 0]
    assert PROXY_FEATURE not in frame.columns


def test_outflow_proxy_rejects_missing_inputs() -> None:
    with pytest.raises(ValueError, match="missing"):
        add_deployment_outflow_proxy(pd.DataFrame({"signer_lamport_delta": [-100]}))
    with pytest.raises(ValueError, match="missing values"):
        add_deployment_outflow_proxy(
            pd.DataFrame(
                {
                    "signer_lamport_delta": [-100, None],
                    "tx_fee_lamports": [5, 5],
                }
            )
        )


def test_noninferiority_decision_uses_predeclared_margin() -> None:
    assert noninferiority_decision([0.0, -NONINFERIORITY_MARGIN], -0.001) == (
        "supported_within_predeclared_noninferiority_margin"
    )
    assert noninferiority_decision([0.0, -NONINFERIORITY_MARGIN - 1e-6], 0.0) == (
        "rejected_proxy_exceeds_noninferiority_margin"
    )
    with pytest.raises(ValueError, match="at least one"):
        noninferiority_decision([], 0.0)
