"""Fee-adjusted target-wallet PnL and validation-window replica comparison."""

import hashlib
import json

import numpy as np
import pandas as pd

from solana_sniper.manifest import append_experiment, git_head, sha256_file
from solana_sniper.paths import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, REPORT_DIR, project_relative
from solana_sniper.replica_backtest import JUNE_START
from solana_sniper.splits import chronological_train_validation_test_split

WALLET_PATH = RAW_DIR / "wallet" / "5brv79e_activity.parquet"
ENTRY_LATENCY_PATH = PROCESSED_DIR / "entry_latency.parquet"
CLASSIFICATION_PATH = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
CLASSIFICATION_MANIFEST_PATH = PROCESSED_DIR / "creator_history_manifest.json"
REPLICA_METRICS_PATH = REPORT_DIR / "position_lag_validation_backtest.json"
METRICS_PATH = REPORT_DIR / "competitor_fee_pnl.json"
REPORT_PATH = REPORT_DIR.parent / "competitor_fee_pnl.md"
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


def render_comparison_svg(comparison: dict[str, object]) -> str:
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
    return "\n".join(elements) + "\n"


def run_competitor_pnl() -> dict[str, object]:
    for path in (
        WALLET_PATH,
        ENTRY_LATENCY_PATH,
        CLASSIFICATION_PATH,
        CLASSIFICATION_MANIFEST_PATH,
        REPLICA_METRICS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    cashflows, cashflow_audit = build_token_cashflows(pd.read_parquet(WALLET_PATH))
    validate_cashflows(cashflows)

    classification = pd.read_parquet(CLASSIFICATION_PATH)
    if classification.empty or classification["token_address"].duplicated().any():
        raise ValueError("classification tokens must be nonempty and unique")
    classification_sha256 = sha256_file(CLASSIFICATION_PATH)
    history_manifest = json.loads(CLASSIFICATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if history_manifest["sha256"] != classification_sha256:
        raise ValueError("strict-history manifest hash does not match classification dataset")
    if history_manifest["rows"] != len(classification):
        raise ValueError("strict-history manifest row count does not match")
    if history_manifest["unique_tokens"] != classification["token_address"].nunique():
        raise ValueError("strict-history manifest unique-token count does not match")
    if history_manifest["strict_time_violations"] != 0:
        raise ValueError("strict-history manifest reports time-boundary violations")
    canonical_time = pd.to_datetime(classification["decision_time"], utc=True)
    utc_hour_mismatches = int(classification["decision_hour_utc"].ne(canonical_time.dt.hour).sum())
    utc_weekday_mismatches = int(
        classification["decision_weekday_utc"].ne(canonical_time.dt.dayofweek).sum()
    )
    if utc_hour_mismatches or utc_weekday_mismatches:
        raise ValueError("classification dataset contains noncanonical UTC clock features")
    positive_times = classification.loc[classification["label"] == 1, "decision_time"]
    active = classification.loc[
        classification["decision_time"].between(positive_times.min(), positive_times.max())
    ].copy()
    split = chronological_train_validation_test_split(
        active,
        time_column="decision_time",
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    holdout_start = split.test["decision_time"].min()
    if not split.validation["decision_time"].max() < holdout_start:
        raise AssertionError("validation overlaps the final holdout")
    latency = pd.read_parquet(ENTRY_LATENCY_PATH)[["token_address", "decision_time"]]
    if latency.empty or latency["token_address"].duplicated().any():
        raise ValueError("entry latency must be nonempty and unique by token")
    positive_token_times = classification.loc[classification["label"] == 1].set_index(
        "token_address"
    )["decision_time"]
    latency_token_times = latency.set_index("token_address")["decision_time"]
    latency_time_mismatches = int(
        pd.to_datetime(latency_token_times, utc=True)
        .ne(pd.to_datetime(positive_token_times.loc[latency_token_times.index], utc=True))
        .sum()
    )
    if latency_time_mismatches:
        raise ValueError("entry-latency decision times differ from corrected classification times")
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
    if replica["classification_dataset_sha256"] != classification_sha256:
        raise ValueError("position-lag report is tied to a different classification dataset")
    if replica["final_holdout_evaluated"]:
        raise ValueError("position-lag report unexpectedly evaluates the final holdout")
    if pd.Timestamp(replica["max_backtest_outcome_utc"]) >= holdout_start:
        raise ValueError("position-lag outcome crosses the final holdout boundary")
    replica_median_rows = [
        row for row in replica["backtest_results"] if row["fee_scenario"] == "training_median_fee"
    ]
    if len(replica_median_rows) != 1:
        raise ValueError("position-lag report must contain one training-median-fee row")
    replica_median_fee = replica_median_rows[0]
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
    figure_svg = render_comparison_svg(comparison)

    hypothesis_supported = (
        development_metrics["net_pnl_usd"] > 0
        and development_metrics["net_mean_roi"] > 0
        and development_metrics["net_median_roi"] > 0
    )
    metrics: dict[str, object] = {
        "experiment": "corrected_target_wallet_recorded_fee_pnl",
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
        "classification_dataset_sha256": classification_sha256,
        "classification_manifest": project_relative(CLASSIFICATION_MANIFEST_PATH),
        "classification_manifest_sha256": sha256_file(CLASSIFICATION_MANIFEST_PATH),
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
        "figure_sha256": hashlib.sha256(figure_svg.encode("utf-8")).hexdigest(),
        "schema": {
            "classification_rows": int(len(classification)),
            "classification_active_rows": int(len(active)),
            "classification_unique_tokens": int(classification["token_address"].nunique()),
            "strict_time_violations": int(history_manifest["strict_time_violations"]),
            "utc_hour_mismatch_rows": utc_hour_mismatches,
            "utc_weekday_mismatch_rows": utc_weekday_mismatches,
            "entry_latency_rows": int(len(latency)),
            "entry_latency_unique_tokens": int(latency["token_address"].nunique()),
            "entry_latency_decision_time_mismatch_rows": latency_time_mismatches,
            "wallet_trade_rows": int(cashflow_audit["trade_rows"]),
            "wallet_unique_transaction_hashes": int(cashflow_audit["unique_transaction_hashes"]),
        },
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
    return metrics


def render_report(metrics: dict[str, object]) -> str:
    full = metrics["full_period_target_wallet"]
    comparison = metrics["development_head_to_head"]
    target = comparison["target_wallet"]
    replica = comparison["position_lag_replica"]
    operating = comparison["classification_overlap_operating_point"]
    replica_entries = (
        f"{replica['executed_sample_rows']:,} sampled fills / "
        f"weight {replica['executed_population_weight']:,.0f}"
    )
    return f"""# Corrected target-wallet fee-adjusted PnL and replica comparison

## Cash-flow audit

This run tests whether the target wallet remains profitable after subtracting recorded network
and DEX fees. It does not tune the classifier. Post-deployment cash flows are behavior and
evaluation data only; no realized return, price, trade, or PnL field enters the deployment-time
selector.

The wallet source has {metrics["cashflow_audit"]["trade_rows"]:,} buy/sell rows and the same number
of unique transaction hashes. Net PnL is `sell receipts - buy costs - gas_usd - dex_usd`.
Priority and tip fees are not added separately because they are already components of the recorded
gas field. Closed-token summaries require both buy and sell activity plus an observed close flag.

## Full-period target description

This section is descriptive competitor analysis, not model selection or final-holdout evaluation.

| metric | value |
|:---|---:|
| Matched closed tokens | {full["token_rows"]:,} |
| Gross buy capital | ${full["gross_buy_usd"]:,.0f} |
| Gross PnL | ${full["gross_pnl_usd"]:,.0f} |
| Recorded fees | ${full["total_fee_usd"]:,.0f} |
| Net PnL | ${full["net_pnl_usd"]:,.0f} |
| Net mean / median ROI | {full["net_mean_roi"]:+.2%} / {full["net_median_roi"]:+.2%} |
| Net hit rate | {full["net_hit_rate"]:.2%} |
| Realized max drawdown | {full["max_drawdown_fraction"]:.2%} |

## Strictly pre-holdout comparison

The target window is `{comparison["time_window_start_utc"]}` through
`{comparison["time_window_end_exclusive_utc"]}` (exclusive). All included target outcomes close
before that boundary; {metrics["development_outcome_censored_tokens"]} crossing tokens are
excluded. The corrected classification table has SHA-256
`{metrics["classification_dataset_sha256"]}` and zero UTC clock mismatches and strict-history
violations.

| metric | target wallet, recorded fees | executable position-lag replica, median fees |
|:---|---:|---:|
| Entry rows | {target["token_rows"]:,} actual buys | {replica_entries} |
| Net mean ROI | {target["net_mean_return"]:+.2%} | {replica["net_mean_return"]:+.2%} |
| Net median ROI | {target["net_median_return"]:+.2%} | {replica["net_median_return"]:+.2%} |
| Hit rate | {target["hit_rate"]:.2%} | {replica["hit_rate"]:.2%} |
| Max drawdown | {target["max_drawdown_fraction"]:.2%} | {replica["max_drawdown_fraction"]:.2%} |
| Net PnL | ${target["net_pnl_usd"]:,.0f} actual | ${replica["net_pnl_usd"]:,.0f} weighted proxy |

The overlapping selector operating point is precision {operating["precision"]:.2%}, recall
{operating["recall"]:.2%}, and F1 {operating["f1"]:.3f}. The target is profitable in aggregate,
mean, and median after recorded fees. The corrected position-lag replica has a positive weighted
mean but a negative typical trade and an insolvent tight-capital path, so it is not economically
equivalent to the target.

![Corrected target-versus-replica comparison](figures/competitor_fee_pnl.svg)

## Decision and limitations

Decision: `{metrics["decision"]}`. This supports the target-wallet fee hypothesis, not replica
profitability. Target results use actual variable sizing and actual entries; replica results use a
fixed notional, sampled negatives with population weights, and a transaction-position entry
proxy. Total dollars are not directly comparable. Cost fields are accepted without independent
on-chain reconciliation, and realized drawdown books PnL at the final sell rather than marking
intratrade risk. The classifier and replica final chronological holdout remain sealed.
"""


def main() -> None:
    first = run_competitor_pnl()
    second = run_competitor_pnl()
    if first != second:
        raise AssertionError("deterministic competitor PnL rerun did not match")
    metrics = first
    report = render_report(metrics)
    figure = render_comparison_svg(metrics["development_head_to_head"])
    if hashlib.sha256(figure.encode("utf-8")).hexdigest() != metrics["figure_sha256"]:
        raise AssertionError("competitor figure hash changed between validation and write")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")
    FIGURE_PATH.write_text(figure, encoding="utf-8", newline="\n")
    append_experiment(
        {
            **metrics,
            "metrics_path": project_relative(METRICS_PATH),
            "metrics_sha256": sha256_file(METRICS_PATH),
            "report_path": project_relative(REPORT_PATH),
            "report_sha256": sha256_file(REPORT_PATH),
        }
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
