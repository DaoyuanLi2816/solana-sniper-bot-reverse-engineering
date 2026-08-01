import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from solana_sniper.paths import PROCESSED_DIR, RAW_DIR, REPORT_DIR

NUMERIC_COLUMNS = [
    "token_amount",
    "quote_amount",
    "price_usd",
    "cost_usd",
    "buy_cost_usd",
    "gas_usd",
    "dex_usd",
]


def _finite_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.9)),
    }


def build_wallet_audit(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    frame = frame.copy()
    frame["event_time"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    trades = frame[frame["event_type"].isin(["buy", "sell"])].copy()
    buys = trades[trades["event_type"] == "buy"].copy()
    sells = trades[trades["event_type"] == "sell"].copy()

    per_token = trades.groupby("token_address", dropna=False).agg(
        token_symbol=("token_symbol", "first"),
        first_event_time=("event_time", "min"),
        last_event_time=("event_time", "max"),
        buy_transactions=("event_type", lambda value: int((value == "buy").sum())),
        sell_transactions=("event_type", lambda value: int((value == "sell").sum())),
    )
    buy_cash = buys.groupby("token_address")["cost_usd"].sum(min_count=1).rename("gross_buy_usd")
    sell_cash = sells.groupby("token_address")["cost_usd"].sum(min_count=1).rename("gross_sell_usd")
    per_token = per_token.join(buy_cash).join(sell_cash).reset_index()
    per_token["gross_pnl_usd"] = per_token["gross_sell_usd"] - per_token["gross_buy_usd"]
    per_token["hold_seconds"] = (
        per_token["last_event_time"] - per_token["first_event_time"]
    ).dt.total_seconds()

    first_buys = buys.sort_values("event_time", kind="stable").drop_duplicates(
        "token_address", keep="first"
    )
    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "row_count": int(len(frame)),
        "transaction_count": int(frame["tx_hash"].nunique()),
        "event_type_counts": {
            str(key): int(value) for key, value in frame["event_type"].value_counts().items()
        },
        "time_start_utc": frame["event_time"].min().isoformat(),
        "time_end_utc": frame["event_time"].max().isoformat(),
        "unique_tokens_traded": int(trades["token_address"].nunique()),
        "unique_tokens_bought": int(buys["token_address"].nunique()),
        "entry_size_usd": _finite_summary(first_buys["cost_usd"]),
        "all_buy_size_usd": _finite_summary(buys["cost_usd"]),
        "sell_transactions_per_token": _finite_summary(per_token["sell_transactions"]),
        "hold_seconds": _finite_summary(per_token["hold_seconds"]),
        "gross_token_pnl_usd": _finite_summary(per_token["gross_pnl_usd"]),
        "tokens_with_multiple_sells": int((per_token["sell_transactions"] > 1).sum()),
        "tokens_without_observed_sell": int((per_token["sell_transactions"] == 0).sum()),
        "limitations": [
            "Wallet-level cash flow is gross and does not yet reconstruct inventory lots.",
            "Deployment latency requires joining deployment and buy transactions.",
            "Post-deployment outcomes are evaluation inputs, never classifier features.",
        ],
    }
    return summary, per_token


def main() -> None:
    source = RAW_DIR / "wallet" / "5brv79e_activity.parquet"
    frame = pq.read_table(source).to_pandas()
    summary, per_token = build_wallet_audit(frame)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    per_token.to_parquet(PROCESSED_DIR / "wallet_token_summary.parquet", index=False)
    report_path = REPORT_DIR / "wallet_audit.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
