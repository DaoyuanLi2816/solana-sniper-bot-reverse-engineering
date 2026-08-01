import pandas as pd
import pytest

from solana_sniper.guardrails import (
    assert_feature_names_are_pre_decision,
    assert_history_precedes_decision,
)


def test_rejects_explicit_post_decision_feature() -> None:
    with pytest.raises(ValueError, match="Post-decision"):
        assert_feature_names_are_pre_decision(["creator_age", "future_return_5m"])


def test_accepts_pre_decision_feature_names() -> None:
    assert_feature_names_are_pre_decision(["creator_prior_deploy_count", "dev_buy_lamports"])


def test_rejects_future_history() -> None:
    history = pd.Series(pd.to_datetime(["2026-01-02", "2026-01-04"], utc=True))
    decision = pd.Series(pd.to_datetime(["2026-01-03", "2026-01-03"], utc=True))
    with pytest.raises(ValueError, match="future-history"):
        assert_history_precedes_decision(history, decision)
