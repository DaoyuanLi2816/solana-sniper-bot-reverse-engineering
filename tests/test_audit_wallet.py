import pandas as pd

from solana_sniper.audit_wallet import build_wallet_audit


def test_wallet_audit_counts_partial_exits() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "event_type": ["buy", "sell", "sell"],
            "tx_hash": ["a", "b", "c"],
            "token_address": ["token", "token", "token"],
            "token_symbol": ["T", "T", "T"],
            "token_amount": [10, 5, 5],
            "quote_amount": [1, 1, 2],
            "price_usd": [1, 1, 1],
            "cost_usd": [100, 70, 60],
            "buy_cost_usd": [100, None, None],
            "gas_usd": [1, 1, 1],
            "dex_usd": [1, 1, 1],
        }
    )
    summary, per_token = build_wallet_audit(frame)
    assert summary["tokens_with_multiple_sells"] == 1
    assert per_token.loc[0, "gross_pnl_usd"] == 30
