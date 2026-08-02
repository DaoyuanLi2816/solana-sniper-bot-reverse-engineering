import pandas as pd
import pytest

from solana_sniper.competitor_pnl import (
    build_token_cashflows,
    closed_token_rows,
    summarize_closed_portfolio,
    validate_cashflows,
)


def wallet_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "token_address": "a",
                "timestamp": 100,
                "event_type": "buy",
                "tx_hash": "a1",
                "cost_usd": 100,
                "gas_usd": 2,
                "dex_usd": 1,
                "gas_native": 0.03,
                "priority_fee": 0.02,
                "tip_fee": 0,
                "token_amount": 10,
                "is_open_or_close": 1,
            },
            {
                "token_address": "a",
                "timestamp": 106,
                "event_type": "sell",
                "tx_hash": "a2",
                "cost_usd": 120,
                "gas_usd": 1,
                "dex_usd": 1,
                "gas_native": 0.01,
                "priority_fee": 0,
                "tip_fee": 0,
                "token_amount": 10,
                "is_open_or_close": 1,
            },
            {
                "token_address": "b",
                "timestamp": 110,
                "event_type": "buy",
                "tx_hash": "b1",
                "cost_usd": 50,
                "gas_usd": 1,
                "dex_usd": 0,
                "gas_native": 0.01,
                "priority_fee": 0,
                "tip_fee": 0,
                "token_amount": 5,
                "is_open_or_close": 1,
            },
        ]
    )


def test_cashflows_subtract_recorded_fees_once() -> None:
    cashflows, audit = build_token_cashflows(wallet_fixture())
    validate_cashflows(cashflows)
    row = cashflows.loc[cashflows["token_address"] == "a"].iloc[0]

    assert audit["trade_rows"] == audit["unique_transaction_hashes"] == 3
    assert row["gross_pnl_usd"] == 20
    assert row["network_fee_usd"] == 3
    assert row["dex_fee_usd"] == 2
    assert row["net_pnl_usd"] == 15
    assert row["net_roi"] == 0.15
    assert row["hold_seconds"] == 6
    assert len(closed_token_rows(cashflows)) == 1


def test_portfolio_summary_reports_realized_drawdown() -> None:
    cashflows, _ = build_token_cashflows(wallet_fixture())
    closed = closed_token_rows(cashflows)
    closed["decision_time"] = pd.Timestamp("2026-01-01T00:00:00Z")
    result = summarize_closed_portfolio(closed)

    assert result["token_rows"] == 1
    assert result["net_pnl_usd"] == 15
    assert result["net_hit_rate"] == 1
    assert result["net_mean_roi"] == pytest.approx(0.15)
    assert result["capital_model_initial_usd"] == 103
    assert result["max_drawdown_usd"] == 0
