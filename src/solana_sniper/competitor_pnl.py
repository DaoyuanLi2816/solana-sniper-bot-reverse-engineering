"""Fee-adjusted target-wallet PnL and validation-window replica comparison."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from solana_sniper.manifest import append_experiment, git_head, sha256_file
from solana_sniper.paths import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, REPORT_DIR, project_relative
from solana_sniper.replica_backtest import JUNE_START
from solana_sniper.splits import chronological_train_validation_test_split

WALLET_PATH = RAW_DIR / "wallet" / "5brv79e_activity.parquet"
ENTRY_LATENCY_PATH = PROCESSED_DIR / "entry_latency.parquet"
CLASSIFICATION_PATH = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
REPLICA_METRICS_PATH = REPORT_DIR / "position_lag_validation_backtest.json"
METRICS_PATH = REPORT_DIR / "competitor_fee_pnl.json"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "competitor_fee_pnl.svg"
NUMERIC_COLUMNS = [
    "cost_usd",
    "gas_usd",
    "dex_usd",
    "gas_native",
    "priority_fee",
    "tip_fee",
    "token_amount",
]


def build_token_cashflows(wallet: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate actual wallet cash flows without double-counting priority or tip fees."""
    frame = wallet.loc[wallet["event_type"].isin(["buy", "sell"])].copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["event_time"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    if frame["tx_hash"].duplicated().any():
        raise ValueError("Trade rows are not unique by transaction hash; fee double-count risk")
    if frame[["cost_usd", "gas_usd", "dex_usd"]].isna().any(axis=None):
        raise ValueError("Wallet trade cash-flow columns contain nulls")
    if (frame[["cost_usd", "gas_usd", "dex_usd"]] < 0).any(axis=None):
        raise ValueError("Wallet trade cash-flow columns contain negative values")

    frame["buy_cost_usd_derived"] = np.where(frame["event_type"] == "buy", frame["cost_usd"], 0)
    frame["sell_receipt_usd"] = np.where(frame["event_type"] == "sell", frame["cost_usd"], 0)
    frame["buy_gas_usd"] = np.where(frame["event_type"] == "buy", frame["gas_usd"], 0)
    frame["sell_gas_usd"] = np.where(frame["event_type"] == "sell", frame["gas_usd"], 0)
    frame["buy_dex_usd"] = np.where(frame["event_type"] == "buy", frame["dex_usd"], 0)
    frame["sell_dex_usd"] = np.where(frame["event_type"] == "sell", frame["dex_usd"], 0)
    frame["buy_token_amount"] = np.where(frame["event_type"] == "buy", frame["token_amount"], 0)
    frame["sell_token_amount"] = np.where(frame["event_type"] == "sell", frame["token_amount"], 0)
    frame["closed_sell"] = (frame["event_type"] == "sell") & (frame["is_open_or_close"] == 1)

    cashflows = frame.groupby("token_address", sort=False).agg(
        first_event_time=("event_time", "min"),
        last_event_time=("event_time", "max"),
        first_buy_time=(
            "event_time",
            lambda value: value[frame.loc[value.index, "event_type"] == "buy"].min(),
        ),
        last_sell_time=(
            "event_time",
            lambda value: value[frame.loc[value.index, "event_type"] == "sell"].max(),
        ),
        buy_transactions=("event_type", lambda value: int((value == "buy").sum())),
        sell_transactions=("event_type", lambda value: int((value == "sell").sum())),
        observed_close=("closed_sell", "any"),
        gross_buy_usd=("buy_cost_usd_derived", "sum"),
        gross_sell_usd=("sell_receipt_usd", "sum"),
        buy_gas_usd=("buy_gas_usd", "sum"),
        sell_gas_usd=("sell_gas_usd", "sum"),
        buy_dex_usd=("buy_dex_usd", "sum"),
        sell_dex_usd=("sell_dex_usd", "sum"),
        bought_token_amount=("buy_token_amount", "sum"),
        sold_token_amount=("sell_token_amount", "sum"),
    )
    cashflows["network_fee_usd"] = cashflows["buy_gas_usd"] + cashflows["sell_gas_usd"]
    cashflows["dex_fee_usd"] = cashflows["buy_dex_usd"] + cashflows["sell_dex_usd"]
    cashflows["total_fee_usd"] = cashflows["network_fee_usd"] + cashflows["dex_fee_usd"]
    cashflows["gross_pnl_usd"] = cashflows["gross_sell_usd"] - cashflows["gross_buy_usd"]
    cashflows["net_pnl_usd"] = cashflows["gross_pnl_usd"] - cashflows["total_fee_usd"]
    cashflows["gross_roi"] = cashflows["gross_pnl_usd"] / cashflows["gross_buy_usd"]
    cashflows["net_roi"] = cashflows["net_pnl_usd"] / cashflows["gross_buy_usd"]
    cashflows["hold_seconds"] = (
        cashflows["last_sell_time"] - cashflows["first_buy_time"]
    ).dt.total_seconds()
    cashflows["inventory_exit_ratio"] = (
        cashflows["sold_token_amount"] / cashflows["bought_token_amount"]
    )
    cashflows = cashflows.reset_index()

    native_fee_residual = (
        frame["gas_native"].fillna(0) - frame["priority_fee"].fillna(0) - frame["tip_fee"].fillna(0)
    )
    audit = {
        "trade_rows": int(len(frame)),
        "unique_transaction_hashes": int(frame["tx_hash"].nunique()),
        "duplicate_transaction_hash_rows": 0,
        "token_rows": int(len(cashflows)),
        "fee_formula": "gas_usd_plus_dex_usd_priority_and_tip_not_added_again",
        "gas_native_minus_priority_minus_tip": {
            "minimum": float(native_fee_residual.min()),
            "median": float(native_fee_residual.median()),
            "maximum": float(native_fee_residual.max()),
            "negative_rows": int((native_fee_residual < -1e-12).sum()),
        },
    }
    return cashflows, audit


def validate_cashflows(frame: pd.DataFrame) -> None:
    if frame.empty or frame["token_address"].duplicated().any():
        raise ValueError("Token cash flows must be nonempty and unique by token")
    if (frame["first_event_time"] > frame["last_event_time"]).any():
        raise ValueError("Token event times are reversed")
    bought = frame.loc[frame["buy_transactions"] > 0]
    if (bought["gross_buy_usd"] <= 0).any():
        raise ValueError("Bought tokens contain nonpositive gross buy cost")
    finite_columns = [
        "gross_buy_usd",
        "gross_sell_usd",
        "network_fee_usd",
        "dex_fee_usd",
        "gross_pnl_usd",
        "net_pnl_usd",
    ]
    if not np.isfinite(frame[finite_columns].to_numpy(dtype=float)).all():
        raise ValueError("Token cash flows contain nonfinite values")


def closed_token_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        (frame["buy_transactions"] > 0)
        & (frame["sell_transactions"] > 0)
        & frame["observed_close"]
        & frame["gross_buy_usd"].gt(0)
    ].copy()


def _realized_drawdown(frame: pd.DataFrame) -> dict[str, float | bool]:
    events: list[tuple[pd.Timestamp, int, float]] = []
    entry_capital = frame["gross_buy_usd"] + frame["buy_gas_usd"] + frame["buy_dex_usd"]
    for first_buy, last_sell, capital in zip(
        frame["first_buy_time"], frame["last_sell_time"], entry_capital, strict=True
    ):
        events.append((first_buy, 1, float(capital)))
        events.append((last_sell, 0, -float(capital)))
    concurrent = 0.0
    maximum_concurrent = 0.0
    for _, _, change in sorted(events):
        concurrent += change
        maximum_concurrent = max(maximum_concurrent, concurrent)

    timeline = frame.groupby("last_sell_time", sort=True)["net_pnl_usd"].sum()
    equity = maximum_concurrent + timeline.cumsum().to_numpy()
    peaks = np.maximum.accumulate(np.concatenate(([maximum_concurrent], equity)))
    drawdowns = peaks[1:] - equity
    index = int(np.argmax(drawdowns))
    maximum_drawdown = float(drawdowns[index])
    return {
        "capital_model_initial_usd": float(maximum_concurrent),
        "ending_equity_usd": float(equity[-1]),
        "minimum_equity_usd": float(equity.min()),
        "insolvent_under_capital_model": bool(equity.min() < 0),
        "max_drawdown_usd": maximum_drawdown,
        "max_drawdown_fraction": float(maximum_drawdown / peaks[index + 1]),
    }


def summarize_closed_portfolio(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        raise ValueError("Closed-token portfolio is empty")
    wins = frame.loc[frame["net_pnl_usd"] > 0, "net_pnl_usd"]
    losses = frame.loc[frame["net_pnl_usd"] <= 0, "net_pnl_usd"]
    return {
        "token_rows": int(len(frame)),
        "decision_time_start_utc": frame["decision_time"].min().isoformat()
        if "decision_time" in frame
        else None,
        "decision_time_end_utc": frame["decision_time"].max().isoformat()
        if "decision_time" in frame
        else None,
        "outcome_time_end_utc": frame["last_sell_time"].max().isoformat(),
        "gross_buy_usd": float(frame["gross_buy_usd"].sum()),
        "gross_sell_usd": float(frame["gross_sell_usd"].sum()),
        "gross_pnl_usd": float(frame["gross_pnl_usd"].sum()),
        "network_fee_usd": float(frame["network_fee_usd"].sum()),
        "dex_fee_usd": float(frame["dex_fee_usd"].sum()),
        "total_fee_usd": float(frame["total_fee_usd"].sum()),
        "net_pnl_usd": float(frame["net_pnl_usd"].sum()),
        "gross_mean_roi": float(frame["gross_roi"].mean()),
        "gross_median_roi": float(frame["gross_roi"].median()),
        "net_mean_roi": float(frame["net_roi"].mean()),
        "net_median_roi": float(frame["net_roi"].median()),
        "net_notional_weighted_roi": float(
            frame["net_pnl_usd"].sum() / frame["gross_buy_usd"].sum()
        ),
        "net_hit_rate": float((frame["net_pnl_usd"] > 0).mean()),
        "average_win_usd": float(wins.mean()),
        "average_loss_usd": float(losses.mean()),
        "net_pnl_usd_quantiles": {
            "p10": float(frame["net_pnl_usd"].quantile(0.1)),
            "median": float(frame["net_pnl_usd"].median()),
            "p90": float(frame["net_pnl_usd"].quantile(0.9)),
            "p99": float(frame["net_pnl_usd"].quantile(0.99)),
        },
        "hold_seconds": {
            "median": float(frame["hold_seconds"].median()),
            "p90": float(frame["hold_seconds"].quantile(0.9)),
        },
        "sell_transactions": {
            "median": float(frame["sell_transactions"].median()),
            "p90": float(frame["sell_transactions"].quantile(0.9)),
        },
        "inventory_exit_ratio": {
            "median": float(frame["inventory_exit_ratio"].median()),
            "p10": float(frame["inventory_exit_ratio"].quantile(0.1)),
            "p90": float(frame["inventory_exit_ratio"].quantile(0.9)),
        },
        **_realized_drawdown(frame),
    }


def write_comparison_svg(comparison: dict[str, object], path: Path) -> None:
    target = comparison["target_wallet"]
    replica = comparison["position_lag_replica"]
    rows = [
        ("Net mean ROI", float(target["net_mean_return"]), float(replica["net_mean_return"])),
        ("Net median ROI", float(target["net_median_return"]), float(replica["net_median_return"])),
        ("Hit rate", float(target["hit_rate"]), float(replica["hit_rate"])),
    ]
    scale = max(max(abs(value) for _, left, right in rows for value in (left, right)), 0.01)
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="430" viewBox="0 0 860 430">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="38" font-family="Arial" font-size="22" font-weight="700" '
        'fill="#0f172a">Development-window target versus replica</text>',
        '<text x="32" y="62" font-family="Arial" font-size="13" fill="#475569">'
        "Recorded target fees versus replica median-fee assumption</text>",
        '<text x="360" y="96" text-anchor="middle" font-family="Arial" font-size="14" '
        'font-weight="700" fill="#334155">Target wallet</text>',
        '<text x="625" y="96" text-anchor="middle" font-family="Arial" font-size="14" '
        'font-weight="700" fill="#334155">Position-lag replica</text>',
    ]
    for index, (label, target_value, replica_value) in enumerate(rows):
        y = 126 + index * 88
        elements.append(
            f'<text x="32" y="{y + 34}" font-family="Arial" font-size="14" '
            f'fill="#334155">{label}</text>'
        )
        for x, value in ((255, target_value), (520, replica_value)):
            color = "#16a34a" if value >= 0 else "#dc2626"
            opacity = 0.2 + 0.6 * min(abs(value) / scale, 1.0)
            elements.extend(
                [
                    f'<rect x="{x}" y="{y}" width="210" height="58" rx="7" '
                    f'fill="{color}" fill-opacity="{opacity:.3f}"/>',
                    f'<text x="{x + 105}" y="{y + 37}" text-anchor="middle" '
                    f'font-family="Arial" font-size="17" font-weight="700" fill="#0f172a">'
                    f"{100 * value:+.2f}%</text>",
                ]
            )
    elements.extend(
        [
            '<text x="32" y="414" font-family="Arial" font-size="12" fill="#64748b">'
            "Different entry sets and sizing: comparison is diagnostic, not causal "
            "attribution.</text>",
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def run_competitor_pnl() -> dict[str, object]:
    for path in (WALLET_PATH, ENTRY_LATENCY_PATH, CLASSIFICATION_PATH, REPLICA_METRICS_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    cashflows, cashflow_audit = build_token_cashflows(pd.read_parquet(WALLET_PATH))
    validate_cashflows(cashflows)

    classification = pd.read_parquet(CLASSIFICATION_PATH)
    positive_times = classification.loc[classification["label"] == 1, "decision_time"]
    classification = classification.loc[
        classification["decision_time"].between(positive_times.min(), positive_times.max())
    ]
    split = chronological_train_validation_test_split(
        classification,
        time_column="decision_time",
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    holdout_start = split.test["decision_time"].min()
    latency = pd.read_parquet(ENTRY_LATENCY_PATH)[["token_address", "decision_time"]]
    cashflows = cashflows.merge(latency, on="token_address", how="left", validate="one_to_one")

    full_period = closed_token_rows(cashflows.loc[cashflows["decision_time"].notna()])
    development_tokens = cashflows.loc[
        cashflows["decision_time"].between(JUNE_START, holdout_start, inclusive="left")
    ].copy()
    development_tokens["outcome_crosses_holdout"] = (
        development_tokens["last_event_time"] >= holdout_start
    )
    development_safe = closed_token_rows(
        development_tokens.loc[~development_tokens["outcome_crosses_holdout"]]
    )
    if development_safe["last_event_time"].max() >= holdout_start:
        raise ValueError("Development target-wallet outcomes cross the final holdout boundary")

    full_metrics = summarize_closed_portfolio(full_period)
    development_metrics = summarize_closed_portfolio(development_safe)
    replica = json.loads(REPLICA_METRICS_PATH.read_text(encoding="utf-8"))
    replica_median_fee = next(
        row for row in replica["backtest_results"] if row["fee_scenario"] == "training_median_fee"
    )
    comparison = {
        "time_window_start_utc": JUNE_START.isoformat(),
        "time_window_end_exclusive_utc": holdout_start.isoformat(),
        "classification_overlap_operating_point": replica["classifier_operating_point"],
        "target_wallet": {
            "entry_set": "actual_target_wallet_buys",
            "token_rows": int(development_metrics["token_rows"]),
            "net_mean_return": float(development_metrics["net_mean_roi"]),
            "net_median_return": float(development_metrics["net_median_roi"]),
            "hit_rate": float(development_metrics["net_hit_rate"]),
            "max_drawdown_fraction": float(development_metrics["max_drawdown_fraction"]),
            "net_pnl_usd": float(development_metrics["net_pnl_usd"]),
        },
        "position_lag_replica": {
            "entry_set": "model_selected_population_weighted_candidates",
            "executed_sample_rows": int(replica_median_fee["executed_sample_rows"]),
            "executed_population_weight": float(replica_median_fee["executed_population_weight"]),
            "net_mean_return": float(replica_median_fee["net_mean_return"]),
            "net_median_return": float(replica_median_fee["net_median_return_unweighted"]),
            "hit_rate": float(replica_median_fee["net_hit_rate"]),
            "max_drawdown_fraction": float(replica_median_fee["max_drawdown_fraction"]),
            "net_pnl_usd": float(replica_median_fee["total_weighted_pnl_usd"]),
        },
        "comparability_limit": (
            "target uses actual variable sizing and entry set; replica uses fixed notional and "
            "population-weighted sampled candidates, so total PnL is not directly comparable"
        ),
    }
    write_comparison_svg(comparison, FIGURE_PATH)

    hypothesis_supported = (
        development_metrics["net_pnl_usd"] > 0
        and development_metrics["net_mean_roi"] > 0
        and development_metrics["net_median_roi"] > 0
    )
    metrics: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment": "target_wallet_recorded_fee_pnl",
        "single_hypothesis": (
            "the target wallet remains profitable after subtracting recorded gas and DEX fees"
        ),
        "decision": "supported_on_development_and_full_descriptive"
        if hypothesis_supported
        else "rejected_recorded_fees_remove_development_profitability",
        "feature_use": "none_postdeployment_cashflows_are_behavior_and_evaluation_only",
        "replica_final_holdout_evaluated": False,
        "classifier_final_holdout_evaluated": False,
        "full_period_part1_descriptive_only": True,
        "final_holdout_start_utc": holdout_start.isoformat(),
        "wallet_source": project_relative(WALLET_PATH),
        "wallet_source_sha256": sha256_file(WALLET_PATH),
        "entry_latency": project_relative(ENTRY_LATENCY_PATH),
        "entry_latency_sha256": sha256_file(ENTRY_LATENCY_PATH),
        "classification_dataset": project_relative(CLASSIFICATION_PATH),
        "classification_dataset_sha256": sha256_file(CLASSIFICATION_PATH),
        "replica_metrics": project_relative(REPLICA_METRICS_PATH),
        "replica_metrics_sha256": sha256_file(REPLICA_METRICS_PATH),
        "cashflow_audit": cashflow_audit,
        "matched_full_period_closed_tokens": int(len(full_period)),
        "unmatched_wallet_tokens_excluded_from_time_split": int(
            cashflows["decision_time"].isna().sum()
        ),
        "development_candidate_tokens": int(len(development_tokens)),
        "development_outcome_censored_tokens": int(
            development_tokens["outcome_crosses_holdout"].sum()
        ),
        "full_period_target_wallet": full_metrics,
        "development_target_wallet": development_metrics,
        "development_head_to_head": comparison,
        "figure": project_relative(FIGURE_PATH),
        "figure_sha256": sha256_file(FIGURE_PATH),
        "code_parent_commit": git_head(),
        "limitations": [
            "Cost and fee fields are accepted as provided; no independent on-chain cash "
            "reconciliation is available.",
            "Realized drawdown books token PnL at final sell and omits intratrade "
            "mark-to-market drawdown.",
            "Observed-close flags do not guarantee exact zero residual token inventory.",
            "The replica comparison uses different entry sets, sizing, and sampling weights.",
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    metrics["metrics_path"] = project_relative(METRICS_PATH)
    metrics["metrics_sha256"] = sha256_file(METRICS_PATH)
    append_experiment(metrics)
    return metrics


def main() -> None:
    print(json.dumps(run_competitor_pnl(), indent=2))


if __name__ == "__main__":
    main()
