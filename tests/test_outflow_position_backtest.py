import pytest

from solana_sniper.outflow_position_backtest import position_acceptance_decision


def _row(
    fee: str,
    *,
    raw_mean: float,
    proxy_mean: float,
    raw_drawdown: float,
    proxy_drawdown: float,
    proxy_insolvent: bool,
) -> dict:
    return {
        "fee_scenario": fee,
        "proxy_minus_raw_net_mean_return": proxy_mean - raw_mean,
        "proxy_minus_raw_max_drawdown_fraction": proxy_drawdown - raw_drawdown,
        "raw_signer_delta": {
            "net_mean_return": raw_mean,
            "max_drawdown_fraction": raw_drawdown,
        },
        "fee_adjusted_outflow_proxy": {
            "net_mean_return": proxy_mean,
            "max_drawdown_fraction": proxy_drawdown,
            "insolvent_under_capital_model": proxy_insolvent,
        },
    }


def _passing_rows() -> list[dict]:
    return [
        _row(
            "gross",
            raw_mean=0.1,
            proxy_mean=0.2,
            raw_drawdown=0.5,
            proxy_drawdown=0.2,
            proxy_insolvent=False,
        ),
        _row(
            "training_median_fee",
            raw_mean=0.02,
            proxy_mean=0.1,
            raw_drawdown=1.2,
            proxy_drawdown=0.4,
            proxy_insolvent=False,
        ),
        _row(
            "training_p90_fee",
            raw_mean=-0.1,
            proxy_mean=0.02,
            raw_drawdown=2.0,
            proxy_drawdown=0.8,
            proxy_insolvent=False,
        ),
    ]


def test_position_acceptance_requires_all_predeclared_checks() -> None:
    result = position_acceptance_decision(_passing_rows())
    assert result["decision"] == "supported_fee_robust_position_lag_feasibility_on_development"
    assert result["all_checks_passed"]
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("fee", "field", "value"),
    [
        ("training_median_fee", "net_mean_return", -0.01),
        ("training_p90_fee", "net_mean_return", -0.01),
        ("training_median_fee", "insolvent_under_capital_model", True),
        ("training_p90_fee", "insolvent_under_capital_model", True),
    ],
)
def test_position_acceptance_rejects_failed_proxy_gate(
    fee: str, field: str, value: float | bool
) -> None:
    rows = _passing_rows()
    row = next(item for item in rows if item["fee_scenario"] == fee)
    row["fee_adjusted_outflow_proxy"][field] = value
    if field == "net_mean_return":
        row["proxy_minus_raw_net_mean_return"] = value - row["raw_signer_delta"][field]
    result = position_acceptance_decision(rows)
    assert result["decision"] == "rejected_fee_robust_position_lag_feasibility_on_development"


def test_position_acceptance_rejects_nonimproving_median_drawdown() -> None:
    rows = _passing_rows()
    median = next(item for item in rows if item["fee_scenario"] == "training_median_fee")
    median["proxy_minus_raw_max_drawdown_fraction"] = 0.0
    result = position_acceptance_decision(rows)
    assert not result["all_checks_passed"]


def test_position_acceptance_requires_complete_fee_grid() -> None:
    with pytest.raises(ValueError, match="exactly"):
        position_acceptance_decision(_passing_rows()[:2])
