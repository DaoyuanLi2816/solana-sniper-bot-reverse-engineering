"""Validation-only replica backtest with fixed wallet-derived behavior parameters."""

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from solana_sniper.baseline import (
    NON_FEATURE_COLUMNS,
    _best_f1_threshold,
    _fixed_threshold_metrics,
    _population_weights,
)
from solana_sniper.boosting import BOOSTING_PARAMETERS, build_boosting_model
from solana_sniper.guardrails import assert_feature_names_are_pre_decision
from solana_sniper.manifest import append_experiment, git_head, sha256_file
from solana_sniper.paths import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, REPORT_DIR, project_relative
from solana_sniper.splits import chronological_train_validation_test_split

CLASSIFICATION_PATH = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
ENTRY_PRICES_PATH = PROCESSED_DIR / "replica_entry_prices.parquet"
WALLET_PATH = RAW_DIR / "wallet" / "5brv79e_activity.parquet"
TRADES_PATH = RAW_DIR / "june" / "pumpfun_trades.parquet"
METRICS_PATH = REPORT_DIR / "replica_validation_backtest.json"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "replica_validation_backtest.svg"
HOLD_SECONDS = 6
JUNE_START = pd.Timestamp("2026-06-01T00:00:00Z")


def _finite_rates(frame: pd.DataFrame) -> pd.Series:
    costs = pd.to_numeric(frame["cost_usd"], errors="coerce")
    fees = pd.to_numeric(frame["gas_usd"], errors="coerce").fillna(0) + pd.to_numeric(
        frame["dex_usd"], errors="coerce"
    ).fillna(0)
    rates = 10_000 * fees / costs
    return rates.replace([np.inf, -np.inf], np.nan).dropna().loc[lambda value: value >= 0]


def training_behavior_parameters(
    wallet: pd.DataFrame, train_end: pd.Timestamp
) -> dict[str, object]:
    """Derive hold, notional, and fee scenarios using wallet events available by train_end."""
    frame = wallet.copy()
    frame["event_time"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    frame = frame.loc[
        (frame["event_time"] <= train_end) & frame["event_type"].isin(["buy", "sell"])
    ].copy()
    if frame.empty:
        raise ValueError("No training-period wallet trades")

    per_token = frame.groupby("token_address").agg(
        first_event_time=("event_time", "min"),
        last_event_time=("event_time", "max"),
        has_buy=("event_type", lambda value: bool((value == "buy").any())),
        has_sell=("event_type", lambda value: bool((value == "sell").any())),
    )
    closed = per_token.loc[per_token["has_buy"] & per_token["has_sell"]].copy()
    hold_seconds = (closed["last_event_time"] - closed["first_event_time"]).dt.total_seconds()

    buys = frame.loc[frame["event_type"] == "buy"].sort_values("event_time", kind="stable")
    sells = frame.loc[frame["event_type"] == "sell"]
    first_buys = buys.drop_duplicates("token_address", keep="first")
    buy_fee_rates = _finite_rates(buys)
    sell_fee_rates = _finite_rates(sells)
    if hold_seconds.empty or first_buys.empty or buy_fee_rates.empty or sell_fee_rates.empty:
        raise ValueError("Training wallet history is missing behavior parameters")

    median_roundtrip = float(buy_fee_rates.median() + sell_fee_rates.median())
    p90_roundtrip = float(buy_fee_rates.quantile(0.9) + sell_fee_rates.quantile(0.9))
    return {
        "cutoff_utc": train_end.isoformat(),
        "wallet_trade_rows": int(len(frame)),
        "closed_token_rows": int(len(closed)),
        "hold_seconds": {
            "median": float(hold_seconds.median()),
            "mean": float(hold_seconds.mean()),
            "p90": float(hold_seconds.quantile(0.9)),
        },
        "first_buy_notional_usd": {
            "median": float(pd.to_numeric(first_buys["cost_usd"], errors="coerce").median()),
            "rows": int(len(first_buys)),
        },
        "fee_bps": {
            "buy_median": float(buy_fee_rates.median()),
            "sell_median": float(sell_fee_rates.median()),
            "roundtrip_median": median_roundtrip,
            "buy_p90": float(buy_fee_rates.quantile(0.9)),
            "sell_p90": float(sell_fee_rates.quantile(0.9)),
            "roundtrip_p90": p90_roundtrip,
        },
    }


def _relation(path: Path) -> str:
    escaped = path.as_posix().replace("'", "''")
    return f"read_parquet('{escaped}')"


def build_exit_marks(
    entries: pd.DataFrame,
    trades_path: Path,
    *,
    hold_seconds: int,
    outcome_cutoff_epoch: int,
) -> pd.DataFrame:
    """Mark each entry at the last observed trade at or before its fixed exit time."""
    if entries.empty:
        return entries.copy()
    keys = ["token_address", "requested_delay_slots"]
    if entries.duplicated(keys).any():
        raise ValueError("Duplicate entry candidates")

    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '12GB'")
    connection.register("entry_candidates_frame", entries)
    connection.execute("CREATE TEMP TABLE entry_candidates AS SELECT * FROM entry_candidates_frame")
    trades = _relation(trades_path)
    connection.execute(
        f"""
        CREATE TEMP TABLE relevant_trades AS
        SELECT
            t.token_address,
            t.slot_index_id,
            t.block_slot,
            t.tx_index,
            t.event_index,
            t.block_time,
            t.timestamp,
            t.tx_hash,
            t.side,
            t.program,
            t.price_usd,
            t.price_sol
        FROM {trades} AS t
        SEMI JOIN (SELECT DISTINCT token_address FROM entry_candidates) AS e
          ON t.token_address = e.token_address
        WHERE t.block_time < {outcome_cutoff_epoch}
        """
    )
    exit_ids = connection.execute(
        f"""
        SELECT
            e.token_address,
            e.requested_delay_slots,
            max(t.slot_index_id) AS exit_slot_index_id
        FROM entry_candidates AS e
        JOIN relevant_trades AS t
          ON t.token_address = e.token_address
         AND t.slot_index_id >= e.entry_slot_index_id
         AND t.block_time <= e.entry_block_time + {hold_seconds}
        GROUP BY e.token_address, e.requested_delay_slots
        """
    ).fetchdf()
    if len(exit_ids) != len(entries):
        raise ValueError("Every covered entry should have at least its entry trade as an exit mark")
    connection.register("exit_ids_frame", exit_ids)
    marks = connection.execute(
        """
        SELECT
            x.token_address,
            x.requested_delay_slots,
            x.exit_slot_index_id,
            t.block_slot AS exit_block_slot,
            t.tx_index AS exit_tx_index,
            t.event_index AS exit_event_index,
            t.block_time AS exit_block_time,
            t.timestamp AS exit_timestamp,
            t.tx_hash AS exit_tx_hash,
            t.side AS exit_side,
            t.program AS exit_program,
            t.price_usd AS exit_price_usd,
            t.price_sol AS exit_price_sol
        FROM exit_ids_frame AS x
        JOIN relevant_trades AS t ON t.slot_index_id = x.exit_slot_index_id
        """
    ).fetchdf()
    connection.close()
    if marks.duplicated(keys).any() or len(marks) != len(entries):
        raise ValueError("Exit marks are not one-to-one with entries")
    return entries.merge(marks, on=keys, how="inner", validate="one_to_one")


def calculate_portfolio_metrics(
    trades: pd.DataFrame, *, fee_bps: float, notional_usd: float
) -> dict[str, float | int]:
    """Calculate weighted fixed-notional PnL and capital-aware maximum drawdown."""
    if trades.empty:
        raise ValueError("Backtest subset is empty")
    weights = trades["population_weight"].to_numpy(dtype=float)
    gross_returns = trades["gross_return"].to_numpy(dtype=float)
    net_returns = gross_returns - fee_bps / 10_000
    weighted_pnl = weights * notional_usd * net_returns

    events: list[tuple[int, int, float]] = []
    for entry_time, exit_time, weight in zip(
        trades["entry_block_time"], trades["exit_target_time"], weights, strict=True
    ):
        events.append((int(entry_time), 1, float(weight)))
        events.append((int(exit_time), 0, -float(weight)))
    concurrent = 0.0
    max_concurrent = 0.0
    for _, _, change in sorted(events):
        concurrent += change
        max_concurrent = max(max_concurrent, concurrent)
    initial_capital = notional_usd * max_concurrent

    pnl_timeline = (
        pd.DataFrame(
            {"exit_time": trades["exit_target_time"].to_numpy(), "weighted_pnl": weighted_pnl}
        )
        .groupby("exit_time", sort=True)["weighted_pnl"]
        .sum()
    )
    equity = initial_capital + pnl_timeline.cumsum().to_numpy()
    peaks = np.maximum.accumulate(np.concatenate(([initial_capital], equity)))
    drawdowns = peaks[1:] - equity
    max_drawdown_index = int(np.argmax(drawdowns))
    max_drawdown_usd = float(drawdowns[max_drawdown_index])
    max_drawdown_fraction = float(max_drawdown_usd / peaks[max_drawdown_index + 1])
    ending_equity = float(equity[-1])
    minimum_equity = float(equity.min())

    return {
        "executed_sample_rows": int(len(trades)),
        "executed_population_weight": float(weights.sum()),
        "gross_mean_return": float(np.average(gross_returns, weights=weights)),
        "gross_median_return_unweighted": float(np.median(gross_returns)),
        "gross_p90_return_unweighted": float(np.quantile(gross_returns, 0.9)),
        "gross_p99_return_unweighted": float(np.quantile(gross_returns, 0.99)),
        "net_mean_return": float(np.average(net_returns, weights=weights)),
        "net_median_return_unweighted": float(np.median(net_returns)),
        "net_hit_rate": float(np.average(net_returns > 0, weights=weights)),
        "total_weighted_notional_usd": float(weights.sum() * notional_usd),
        "total_weighted_pnl_usd": float(weighted_pnl.sum()),
        "max_concurrent_weighted_positions": float(max_concurrent),
        "initial_capital_usd": float(initial_capital),
        "ending_equity_usd": ending_equity,
        "minimum_equity_usd": minimum_equity,
        "insolvent_under_capital_model": bool(minimum_equity < 0),
        "max_drawdown_usd": max_drawdown_usd,
        "max_drawdown_fraction": max_drawdown_fraction,
    }


def validate_backtest_rows(frame: pd.DataFrame, outcome_cutoff_epoch: int) -> None:
    keys = ["token_address", "requested_delay_slots"]
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("Backtest rows must be nonempty and unique by token-delay")
    if (frame["entry_block_time"] + HOLD_SECONDS >= outcome_cutoff_epoch).any():
        raise ValueError("Entry hold window crosses the final holdout boundary")
    if (frame["exit_block_time"] > frame["exit_target_time"]).any():
        raise ValueError("Exit mark occurs after the fixed exit time")
    if (frame["exit_slot_index_id"] < frame["entry_slot_index_id"]).any():
        raise ValueError("Exit mark occurs before entry")
    if (frame[["entry_price_usd", "exit_price_usd"]] <= 0).any(axis=None):
        raise ValueError("Backtest contains nonpositive prices")
    if not np.isfinite(frame["gross_return"]).all():
        raise ValueError("Backtest returns are not finite")


def write_backtest_svg(results: list[dict[str, object]], path: Path) -> None:
    rows = [row for row in results if row["execution_policy"] == "all_observed_proxy"]
    fee_order = ["gross", "training_median_fee", "training_p90_fee"]
    by_key = {(str(row["fee_scenario"]), int(row["requested_delay_slots"])): row for row in rows}
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="840" height="410" viewBox="0 0 840 410">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="32" y="38" font-family="Arial" font-size="22" font-weight="700" '
        'fill="#0f172a">Validation replica: weighted mean return</text>',
        '<text x="32" y="62" font-family="Arial" font-size="13" fill="#475569">'
        "Fixed 6-second mark; final chronological holdout not evaluated</text>",
    ]
    for column, delay in enumerate((0, 1, 2)):
        x = 275 + column * 170
        elements.append(
            f'<text x="{x}" y="98" text-anchor="middle" font-family="Arial" '
            f'font-size="15" font-weight="700" fill="#334155">delay {delay}</text>'
        )
    labels = {
        "gross": "Gross",
        "training_median_fee": "Training median fees",
        "training_p90_fee": "Training p90 fees",
    }
    values = [float(row["net_mean_return"]) for row in rows]
    scale = max(max(abs(value) for value in values), 0.01)
    for row_index, fee_name in enumerate(fee_order):
        y = 135 + row_index * 82
        elements.append(
            f'<text x="32" y="{y + 31}" font-family="Arial" font-size="14" '
            f'fill="#334155">{labels[fee_name]}</text>'
        )
        for column, delay in enumerate((0, 1, 2)):
            value = float(by_key[(fee_name, delay)]["net_mean_return"])
            intensity = min(abs(value) / scale, 1.0)
            color = "#16a34a" if value >= 0 else "#dc2626"
            opacity = 0.18 + 0.62 * intensity
            x = 200 + column * 170
            elements.extend(
                [
                    f'<rect x="{x}" y="{y}" width="150" height="54" rx="7" '
                    f'fill="{color}" fill-opacity="{opacity:.3f}"/>',
                    f'<text x="{x + 75}" y="{y + 34}" text-anchor="middle" '
                    f'font-family="Arial" font-size="16" font-weight="700" fill="#0f172a">'
                    f"{100 * value:+.2f}%</text>",
                ]
            )
    elements.extend(
        [
            '<text x="32" y="389" font-family="Arial" font-size="12" fill="#64748b">'
            "Population weighting expands each sampled negative by 25x. "
            "Fees are wallet-training-only.</text>",
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def run_validation_backtest() -> dict[str, object]:
    for path in (CLASSIFICATION_PATH, ENTRY_PRICES_PATH, WALLET_PATH, TRADES_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    frame = pd.read_parquet(CLASSIFICATION_PATH)
    positive_times = frame.loc[frame["label"] == 1, "decision_time"]
    frame = frame.loc[frame["decision_time"].between(positive_times.min(), positive_times.max())]
    numeric_features = [
        column
        for column in frame.select_dtypes(include=["number", "bool"]).columns
        if column not in NON_FEATURE_COLUMNS
    ]
    assert_feature_names_are_pre_decision(numeric_features)
    split = chronological_train_validation_test_split(
        frame, time_column="decision_time", validation_fraction=0.2, test_fraction=0.2
    )
    train_end = split.train["decision_time"].max()
    final_holdout_start = split.test["decision_time"].min()
    outcome_cutoff_epoch = int(final_holdout_start.timestamp())

    model = build_boosting_model()
    model.fit(split.train[numeric_features], split.train["label"])
    probabilities = model.predict_proba(split.validation[numeric_features])[:, 1]
    weights = _population_weights(split.validation["label"])
    operating_point = _best_f1_threshold(split.validation["label"], probabilities, weights)
    classifier_metrics = _fixed_threshold_metrics(
        split.validation["label"], probabilities, operating_point["threshold"], weights
    )

    validation = split.validation[["token_address", "decision_time", "label"]].copy()
    validation["probability"] = probabilities
    validation["population_weight"] = weights
    development = validation.loc[
        (validation["decision_time"] >= JUNE_START)
        & (validation["decision_time"] < final_holdout_start)
    ].copy()
    selected = development.loc[development["probability"] >= operating_point["threshold"]].copy()
    if selected.empty or selected["decision_time"].max() >= final_holdout_start:
        raise ValueError("Selected validation population violates the holdout boundary")

    behavior = training_behavior_parameters(pd.read_parquet(WALLET_PATH), train_end)
    if behavior["hold_seconds"]["median"] != HOLD_SECONDS:
        raise ValueError("Training-only median hold does not match the frozen six-second rule")
    entry_prices = pd.read_parquet(ENTRY_PRICES_PATH)
    attempts = selected.merge(entry_prices, on="token_address", how="left", validate="one_to_many")
    expected_attempts = len(selected) * 3
    if len(attempts) != expected_attempts:
        raise ValueError(f"Expected {expected_attempts} token-delay attempts, got {len(attempts)}")
    attempts["outcome_eligible"] = (
        attempts["covered"]
        & attempts["entry_block_time"].notna()
        & (attempts["entry_block_time"] + HOLD_SECONDS < outcome_cutoff_epoch)
    )
    eligible = attempts.loc[attempts["outcome_eligible"]].copy()
    eligible["exit_target_time"] = eligible["entry_block_time"].astype("int64") + HOLD_SECONDS
    marked = build_exit_marks(
        eligible,
        TRADES_PATH,
        hold_seconds=HOLD_SECONDS,
        outcome_cutoff_epoch=outcome_cutoff_epoch,
    )
    marked["gross_return"] = marked["exit_price_usd"] / marked["entry_price_usd"] - 1
    marked["mark_staleness_seconds"] = marked["exit_target_time"] - marked["exit_block_time"]
    validate_backtest_rows(marked, outcome_cutoff_epoch)

    fee_scenarios = {
        "gross": 0.0,
        "training_median_fee": behavior["fee_bps"]["roundtrip_median"],
        "training_p90_fee": behavior["fee_bps"]["roundtrip_p90"],
    }
    notional = float(behavior["first_buy_notional_usd"]["median"])
    results: list[dict[str, object]] = []
    for delay in (0, 1, 2):
        delay_attempts = attempts.loc[attempts["requested_delay_slots"] == delay]
        delay_rows = marked.loc[marked["requested_delay_slots"] == delay]
        for policy, policy_rows in (
            ("all_observed_proxy", delay_rows),
            ("exact_target_slot_only", delay_rows.loc[delay_rows["wait_slots_beyond_target"] == 0]),
        ):
            for fee_name, fee_bps in fee_scenarios.items():
                metrics = calculate_portfolio_metrics(
                    policy_rows, fee_bps=float(fee_bps), notional_usd=notional
                )
                results.append(
                    {
                        "requested_delay_slots": delay,
                        "execution_policy": policy,
                        "fee_scenario": fee_name,
                        "fee_bps": float(fee_bps),
                        "attempted_sample_rows": int(len(delay_attempts)),
                        "attempted_population_weight": float(
                            delay_attempts["population_weight"].sum()
                        ),
                        "coverage_rate_sampled": float(len(policy_rows) / len(delay_attempts)),
                        "coverage_rate_population_weighted": float(
                            policy_rows["population_weight"].sum()
                            / delay_attempts["population_weight"].sum()
                        ),
                        "actual_delay_slots": {
                            "median": float(policy_rows["actual_delay_slots"].median()),
                            "p90": float(policy_rows["actual_delay_slots"].quantile(0.9)),
                            "max": int(policy_rows["actual_delay_slots"].max()),
                        },
                        "wait_beyond_target_slots": {
                            "median": float(policy_rows["wait_slots_beyond_target"].median()),
                            "p90": float(policy_rows["wait_slots_beyond_target"].quantile(0.9)),
                            "max": int(policy_rows["wait_slots_beyond_target"].max()),
                        },
                        **metrics,
                    }
                )

    write_backtest_svg(results, FIGURE_PATH)
    metrics: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment": "replica_validation_fixed_training_wallet_hold",
        "single_hypothesis": (
            "the training-only target-wallet median hold of six seconds can serve as a fixed "
            "replica exit rule under 0/1/2-slot entry and observed training fee sensitivity"
        ),
        "development_status": "validation_diagnostic_not_independent_final_estimate",
        "final_holdout_evaluated": False,
        "classification_dataset": project_relative(CLASSIFICATION_PATH),
        "classification_dataset_sha256": sha256_file(CLASSIFICATION_PATH),
        "entry_prices": project_relative(ENTRY_PRICES_PATH),
        "entry_prices_sha256": sha256_file(ENTRY_PRICES_PATH),
        "wallet_source": project_relative(WALLET_PATH),
        "wallet_source_sha256": sha256_file(WALLET_PATH),
        "trades_source": project_relative(TRADES_PATH),
        "trades_source_sha256": sha256_file(TRADES_PATH),
        "model_hyperparameters": BOOSTING_PARAMETERS,
        "feature_names": numeric_features,
        "train_end_utc": train_end.isoformat(),
        "validation_start_utc": split.validation["decision_time"].min().isoformat(),
        "validation_end_utc": split.validation["decision_time"].max().isoformat(),
        "final_holdout_start_utc": final_holdout_start.isoformat(),
        "max_predicted_decision_time_utc": development["decision_time"].max().isoformat(),
        "max_selected_decision_time_utc": selected["decision_time"].max().isoformat(),
        "max_backtest_outcome_epoch": int(marked["exit_block_time"].max()),
        "classifier_validation_metrics": classifier_metrics,
        "classifier_operating_point": operating_point,
        "development_june_rows": int(len(development)),
        "selected_sample_rows": int(len(selected)),
        "selected_population_weight": float(selected["population_weight"].sum()),
        "behavior_parameters": behavior,
        "attempt_rows": int(len(attempts)),
        "covered_and_cutoff_safe_rows": int(len(eligible)),
        "backtest_rows": int(len(marked)),
        "mark_staleness_seconds": {
            "median": float(marked["mark_staleness_seconds"].median()),
            "p90": float(marked["mark_staleness_seconds"].quantile(0.9)),
            "max": int(marked["mark_staleness_seconds"].max()),
        },
        "backtest_results": results,
        "figure": project_relative(FIGURE_PATH),
        "figure_sha256": sha256_file(FIGURE_PATH),
        "code_parent_commit": git_head(),
        "limitations": [
            "The six-second exit is a last-trade mark at or before the target time, "
            "not a proven executable sell fill.",
            "The operating threshold is selected on this validation partition, so this "
            "is not an independent final estimate.",
            "The 25x negative sampling weight estimates population metrics but cannot "
            "recover omitted-token path dependence.",
            "Missing and later-than-target entries are reported explicitly; actual delay "
            "can exceed requested delay.",
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    metrics["metrics_path"] = project_relative(METRICS_PATH)
    metrics["metrics_sha256"] = sha256_file(METRICS_PATH)
    append_experiment(metrics)
    return metrics


def main() -> None:
    print(json.dumps(run_validation_backtest(), indent=2))


if __name__ == "__main__":
    main()
