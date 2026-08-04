import pandas as pd
import pytest

from solana_sniper.replica_backtest import (
    backtest_acceptance_decision,
    calculate_portfolio_metrics,
    training_behavior_parameters,
)


def test_training_behavior_uses_only_events_through_train_end() -> None:
    wallet = pd.DataFrame(
        [
            {
                "token_address": "a",
                "timestamp": 100,
                "event_type": "buy",
                "cost_usd": 100,
                "gas_usd": 1,
                "dex_usd": 1,
            },
            {
                "token_address": "a",
                "timestamp": 106,
                "event_type": "sell",
                "cost_usd": 110,
                "gas_usd": 1,
                "dex_usd": 0.1,
            },
            {
                "token_address": "b",
                "timestamp": 200,
                "event_type": "buy",
                "cost_usd": 10_000,
                "gas_usd": 0,
                "dex_usd": 0,
            },
        ]
    )
    result = training_behavior_parameters(wallet, pd.Timestamp(150, unit="s", tz="UTC"))

    assert result["wallet_trade_rows"] == 2
    assert result["closed_token_rows"] == 1
    assert result["hold_seconds"]["median"] == 6
    assert result["first_buy_notional_usd"]["median"] == 100
    assert result["fee_bps"]["roundtrip_median"] == pytest.approx(300)


def test_portfolio_metrics_include_weighted_pnl_and_drawdown() -> None:
    trades = pd.DataFrame(
        {
            "population_weight": [1.0, 2.0],
            "gross_return": [0.1, -0.2],
            "entry_block_time": [10, 12],
            "exit_target_time": [16, 18],
        }
    )
    result = calculate_portfolio_metrics(trades, fee_bps=100, notional_usd=100)

    assert result["executed_population_weight"] == 3
    assert result["net_mean_return"] == pytest.approx(-0.11)
    assert result["net_hit_rate"] == pytest.approx(1 / 3)
    assert result["total_weighted_pnl_usd"] == pytest.approx(-33)
    assert result["max_concurrent_weighted_positions"] == 3
    assert result["initial_capital_usd"] == 300
    assert result["max_drawdown_usd"] == pytest.approx(42)
    assert result["max_drawdown_fraction"] == pytest.approx(42 / 309)


def _acceptance_row(delay: int, mean: float, median: float, insolvent: bool) -> dict[str, object]:
    return {
        "requested_delay_slots": delay,
        "execution_policy": "all_observed_proxy",
        "fee_scenario": "training_median_fee",
        "net_mean_return": mean,
        "net_median_return_unweighted": median,
        "insolvent_under_capital_model": insolvent,
    }


def test_backtest_acceptance_requires_zero_as_the_only_viable_delay() -> None:
    supported = backtest_acceptance_decision(
        [
            _acceptance_row(0, 0.2, 0.1, False),
            _acceptance_row(1, 0.1, -0.1, False),
            _acceptance_row(2, 0.3, 0.1, True),
        ]
    )
    assert supported["decision"] == "supported_only_requested_delay_zero_is_viable"
    assert supported["observed_viable_delays"] == [0]

    rejected = backtest_acceptance_decision(
        [
            _acceptance_row(0, 0.2, 0.1, False),
            _acceptance_row(1, 0.1, 0.01, False),
            _acceptance_row(2, -0.1, -0.1, False),
        ]
    )
    assert rejected["decision"] == "rejected_viable_delay_set_differs_from_zero_only"
    assert rejected["observed_viable_delays"] == [0, 1]

    with pytest.raises(ValueError, match="exactly"):
        backtest_acceptance_decision([_acceptance_row(0, 0.2, 0.1, False)])
