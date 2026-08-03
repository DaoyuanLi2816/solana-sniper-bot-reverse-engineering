import pandas as pd
import pytest

from solana_sniper.metadata_uri_payload import (
    NONINFERIORITY_MARGIN,
    PAYLOAD_FEATURE,
    add_metadata_uri_payload_length,
    payload_decision,
)


def test_metadata_uri_payload_removes_only_ipfs_scheme_prefix() -> None:
    frame = pd.DataFrame(
        {
            "metadata_uri_length": [0, 48, 55, 80],
            "metadata_uri_is_ipfs": [0, 0, 1, 1],
        }
    )
    result = add_metadata_uri_payload_length(frame)
    assert result[PAYLOAD_FEATURE].tolist() == [0, 48, 48, 73]
    assert PAYLOAD_FEATURE not in frame.columns


def test_metadata_uri_payload_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="missing"):
        add_metadata_uri_payload_length(pd.DataFrame({"metadata_uri_length": [10]}))
    with pytest.raises(ValueError, match="missing values"):
        add_metadata_uri_payload_length(
            pd.DataFrame(
                {
                    "metadata_uri_length": [10, None],
                    "metadata_uri_is_ipfs": [0, 0],
                }
            )
        )
    with pytest.raises(ValueError, match="binary"):
        add_metadata_uri_payload_length(
            pd.DataFrame(
                {
                    "metadata_uri_length": [10],
                    "metadata_uri_is_ipfs": [2],
                }
            )
        )
    with pytest.raises(ValueError, match="payload length"):
        add_metadata_uri_payload_length(
            pd.DataFrame(
                {
                    "metadata_uri_length": [6],
                    "metadata_uri_is_ipfs": [1],
                }
            )
        )


def test_payload_decision_separates_improvement_noninferiority_and_rejection() -> None:
    assert payload_decision([0.001, 0.002, 0.003], 0.001) == (
        "supported_improves_all_development_checks"
    )
    assert payload_decision([0.001, -NONINFERIORITY_MARGIN, 0.0], 0.001) == (
        "retained_semantically_within_noninferiority_margin"
    )
    assert payload_decision([0.001, -NONINFERIORITY_MARGIN - 1e-6, 0.0], 0.001) == (
        "rejected_exceeds_noninferiority_margin"
    )
    with pytest.raises(ValueError, match="at least one"):
        payload_decision([], 0.0)
