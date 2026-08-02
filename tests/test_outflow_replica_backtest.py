import pytest

from solana_sniper.outflow_replica_backtest import backtest_acceptance_decision


def _comparison(delay: int, delta: float, raw_insolvent: bool, proxy_insolvent: bool) -> dict:
    return {
        "requested_delay_slots": delay,
        "proxy_minus_raw_net_mean_return": delta,
        "raw_signer_delta": {"insolvent_under_capital_model": raw_insolvent},
        "fee_adjusted_outflow_proxy": {"insolvent_under_capital_model": proxy_insolvent},
    }


def test_backtest_acceptance_requires_two_improvements_and_no_new_insolvency() -> None:
    result = backtest_acceptance_decision(
        [
            _comparison(0, 0.01, False, False),
            _comparison(1, -0.01, True, False),
            _comparison(2, 0.02, True, True),
        ]
    )
    assert result["decision"] == "supported_on_predeclared_development_backtest_criterion"
    assert result["improved_delays"] == [0, 2]
    assert result["proxy_insolvent_delay_count"] == 1


@pytest.mark.parametrize(
    "rows",
    [
        [
            _comparison(0, 0.01, False, False),
            _comparison(1, -0.01, True, False),
            _comparison(2, -0.01, True, False),
        ],
        [
            _comparison(0, 0.01, False, True),
            _comparison(1, 0.01, False, False),
            _comparison(2, 0.01, False, False),
        ],
        [
            _comparison(0, 0.01, False, False),
            _comparison(1, 0.01, False, True),
            _comparison(2, 0.01, False, False),
        ],
    ],
)
def test_backtest_acceptance_rejects_failed_gate(rows: list[dict]) -> None:
    result = backtest_acceptance_decision(rows)
    assert result["decision"] == "rejected_on_predeclared_development_backtest_criterion"


def test_backtest_acceptance_requires_complete_delay_grid() -> None:
    with pytest.raises(ValueError, match="exactly"):
        backtest_acceptance_decision([_comparison(0, 0.01, False, False)])
