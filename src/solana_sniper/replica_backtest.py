"""Validation-only replica backtest with fixed wallet-derived behavior parameters."""

import json
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
from solana_sniper.entry_prices import validate_entry_prices
from solana_sniper.guardrails import assert_feature_names_are_pre_decision
from solana_sniper.manifest import append_experiment, git_head, sha256_file
from solana_sniper.paths import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, REPORT_DIR, project_relative
from solana_sniper.splits import chronological_train_validation_test_split

CLASSIFICATION_PATH = PROCESSED_DIR / "classification_dataset_creator_history.parquet"
CLASSIFICATION_MANIFEST_PATH = PROCESSED_DIR / "creator_history_manifest.json"
ENTRY_PRICES_PATH = PROCESSED_DIR / "replica_entry_prices.parquet"
ENTRY_PRICES_METRICS_PATH = REPORT_DIR / "entry_prices_metrics.json"
WALLET_PATH = RAW_DIR / "wallet" / "5brv79e_activity.parquet"
TRADES_PATH = RAW_DIR / "june" / "pumpfun_trades.parquet"
METRICS_PATH = REPORT_DIR / "replica_validation_backtest.json"
REPORT_PATH = REPORT_DIR.parent / "replica_validation_backtest.md"
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


def backtest_acceptance_decision(results: list[dict[str, object]]) -> dict[str, object]:
    """Require median-fee viability at requested delay zero and nowhere else."""
    primary = [
        row
        for row in results
        if row["execution_policy"] == "all_observed_proxy"
        and row["fee_scenario"] == "training_median_fee"
    ]
    if {int(row["requested_delay_slots"]) for row in primary} != {0, 1, 2}:
        raise ValueError("primary results must contain exactly the 0/1/2-slot rows")
    viable_delays = [
        int(row["requested_delay_slots"])
        for row in primary
        if float(row["net_mean_return"]) > 0
        and float(row["net_median_return_unweighted"]) > 0
        and not bool(row["insolvent_under_capital_model"])
    ]
    supported = viable_delays == [0]
    return {
        "decision": (
            "supported_only_requested_delay_zero_is_viable"
            if supported
            else "rejected_viable_delay_set_differs_from_zero_only"
        ),
        "viability_definition": (
            "training-median-fee all-observed net weighted mean and unweighted median are "
            "strictly positive and the fixed-notional capital path remains solvent"
        ),
        "required_viable_delays": [0],
        "observed_viable_delays": viable_delays,
    }


def render_backtest_svg(results: list[dict[str, object]]) -> str:
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
    return "\n".join(elements) + "\n"


def run_validation_backtest() -> dict[str, object]:
    required_paths = (
        CLASSIFICATION_PATH,
        CLASSIFICATION_MANIFEST_PATH,
        ENTRY_PRICES_PATH,
        ENTRY_PRICES_METRICS_PATH,
        WALLET_PATH,
        TRADES_PATH,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    frame = pd.read_parquet(CLASSIFICATION_PATH)
    required_columns = {
        "label",
        "decision_time",
        "token_address",
        "tx_hash",
        "decision_hour_utc",
        "decision_weekday_utc",
        "creator_prior_deploy_count",
        "creator_prior_active_slot_count",
        "creator_observed_age_seconds",
        "creator_seconds_since_previous_deploy",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"classification dataset is missing columns: {missing_columns}")
    if frame["token_address"].duplicated().any():
        raise ValueError("classification token_address must be unique")
    if frame["label"].nunique() != 2:
        raise ValueError("classification dataset must contain both classes")
    classification_sha256 = sha256_file(CLASSIFICATION_PATH)
    history_manifest = json.loads(CLASSIFICATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if history_manifest["sha256"] != classification_sha256:
        raise ValueError("strict-history manifest hash does not match classification dataset")
    if history_manifest["rows"] != len(frame):
        raise ValueError("strict-history manifest row count does not match")
    if history_manifest["unique_tokens"] != frame["token_address"].nunique():
        raise ValueError("strict-history manifest unique-token count does not match")
    if history_manifest["strict_time_violations"] != 0:
        raise ValueError("strict-history manifest reports time-boundary violations")
    canonical_time = pd.to_datetime(frame["decision_time"], utc=True)
    utc_hour_mismatches = int(frame["decision_hour_utc"].ne(canonical_time.dt.hour).sum())
    utc_weekday_mismatches = int(
        frame["decision_weekday_utc"].ne(canonical_time.dt.dayofweek).sum()
    )
    if utc_hour_mismatches or utc_weekday_mismatches:
        raise ValueError("classification dataset contains noncanonical UTC clock features")

    positive_times = frame.loc[frame["label"] == 1, "decision_time"]
    active = frame.loc[
        frame["decision_time"].between(positive_times.min(), positive_times.max())
    ].copy()
    numeric_features = [
        column
        for column in active.select_dtypes(include=["number", "bool"]).columns
        if column not in NON_FEATURE_COLUMNS
    ]
    assert_feature_names_are_pre_decision(numeric_features)
    split = chronological_train_validation_test_split(
        active, time_column="decision_time", validation_fraction=0.2, test_fraction=0.2
    )
    for name, partition in (
        ("train", split.train),
        ("validation", split.validation),
        ("final_holdout", split.test),
    ):
        if partition["label"].nunique() != 2:
            raise ValueError(f"{name} partition does not contain both classes")
    train_end = split.train["decision_time"].max()
    final_holdout_start = split.test["decision_time"].min()
    if not split.validation["decision_time"].max() < final_holdout_start:
        raise AssertionError("validation overlaps the final holdout")
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
    validate_entry_prices(entry_prices, int(entry_prices["token_address"].nunique()))
    entry_metrics = json.loads(ENTRY_PRICES_METRICS_PATH.read_text(encoding="utf-8"))
    entry_prices_sha256 = sha256_file(ENTRY_PRICES_PATH)
    if entry_metrics["classification_sha256"] != classification_sha256:
        raise ValueError("entry-price audit is tied to a different classification dataset")
    if entry_metrics["output_sha256"] != entry_prices_sha256:
        raise ValueError("entry-price audit hash does not match the entry-price table")
    if entry_metrics["trades_sha256"] != sha256_file(TRADES_PATH):
        raise ValueError("entry-price audit trades hash does not match the source")
    if entry_metrics["rows"] != len(entry_prices):
        raise ValueError("entry-price audit row count does not match")
    entry_token_times = entry_prices.drop_duplicates("token_address").set_index("token_address")[
        "decision_time"
    ]
    classification_token_times = frame.set_index("token_address")["decision_time"]
    if not entry_token_times.equals(classification_token_times.loc[entry_token_times.index]):
        raise ValueError("entry-price decision times differ from corrected classification times")
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

    acceptance = backtest_acceptance_decision(results)
    metrics: dict[str, object] = {
        "experiment": "corrected_raw_signer_history_replica_validation",
        "single_hypothesis": (
            "after UTC remediation the raw signer-delta plus strict-history selector with the "
            "training-only six-second hold and fees is viable only at requested delay zero"
        ),
        "predeclared_acceptance_criterion": (
            "under all-observed execution and training-median fees, viability requires positive "
            "weighted mean, positive unweighted median, and solvency; the viable delay set must "
            "equal only requested delay zero"
        ),
        "acceptance": acceptance,
        "development_status": "validation_diagnostic_not_independent_final_estimate",
        "final_holdout_evaluated": False,
        "final_holdout_status": "sealed_no_predictions_generated",
        "classification_dataset": project_relative(CLASSIFICATION_PATH),
        "classification_dataset_sha256": classification_sha256,
        "classification_manifest": project_relative(CLASSIFICATION_MANIFEST_PATH),
        "classification_manifest_sha256": sha256_file(CLASSIFICATION_MANIFEST_PATH),
        "entry_prices": project_relative(ENTRY_PRICES_PATH),
        "entry_prices_sha256": entry_prices_sha256,
        "entry_prices_metrics": project_relative(ENTRY_PRICES_METRICS_PATH),
        "entry_prices_metrics_sha256": sha256_file(ENTRY_PRICES_METRICS_PATH),
        "wallet_source": project_relative(WALLET_PATH),
        "wallet_source_sha256": sha256_file(WALLET_PATH),
        "trades_source": project_relative(TRADES_PATH),
        "trades_source_sha256": sha256_file(TRADES_PATH),
        "source_role": "post_deployment_prices_used_only_for_backtest_labels_and_marks",
        "model_hyperparameters": BOOSTING_PARAMETERS,
        "feature_names": numeric_features,
        "train_end_utc": train_end.isoformat(),
        "validation_start_utc": split.validation["decision_time"].min().isoformat(),
        "validation_end_utc": split.validation["decision_time"].max().isoformat(),
        "final_holdout_start_utc": final_holdout_start.isoformat(),
        "max_predicted_decision_time_utc": development["decision_time"].max().isoformat(),
        "max_selected_decision_time_utc": selected["decision_time"].max().isoformat(),
        "max_backtest_outcome_epoch": int(marked["exit_block_time"].max()),
        "max_backtest_outcome_utc": pd.Timestamp(
            int(marked["exit_block_time"].max()), unit="s", tz="UTC"
        ).isoformat(),
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
        "schema": {
            "classification_rows": int(len(frame)),
            "classification_active_rows": int(len(active)),
            "classification_unique_tokens": int(frame["token_address"].nunique()),
            "classification_positive_rows": int((frame["label"] == 1).sum()),
            "classification_negative_rows": int((frame["label"] == 0).sum()),
            "strict_time_violations": int(history_manifest["strict_time_violations"]),
            "utc_hour_mismatch_rows": utc_hour_mismatches,
            "utc_weekday_mismatch_rows": utc_weekday_mismatches,
            "entry_price_rows": int(len(entry_prices)),
            "entry_price_unique_tokens": int(entry_prices["token_address"].nunique()),
            "entry_price_unique_token_delay_keys": int(
                entry_prices[["token_address", "requested_delay_slots"]].drop_duplicates().shape[0]
            ),
            "entry_price_decision_time_mismatch_rows": 0,
            "backtest_unique_token_delay_keys": int(
                marked[["token_address", "requested_delay_slots"]].drop_duplicates().shape[0]
            ),
        },
        "code_parent_commit": git_head(),
        "limitations": [
            "The first observed trade at or after each target slot is an optimistic entry proxy.",
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
    return metrics


def render_report(metrics: dict[str, object]) -> str:
    classifier = metrics["classifier_validation_metrics"]
    acceptance = metrics["acceptance"]
    decision_text = (
        "The corrected development result supports the frozen hypothesis: requested delay zero "
        "is the only viable delay under the predeclared definition."
        if acceptance["decision"] == "supported_only_requested_delay_zero_is_viable"
        else "The corrected development result rejects the frozen zero-only viability hypothesis."
    )
    scenario_labels = {
        "gross": "gross (0 bps)",
        "training_median_fee": "training median",
        "training_p90_fee": "training p90",
    }
    policy_labels = {
        "all_observed_proxy": "all observed",
        "exact_target_slot_only": "exact target slot",
    }
    rows = []
    for result in metrics["backtest_results"]:
        if (
            result["execution_policy"] == "exact_target_slot_only"
            and result["fee_scenario"] != "training_median_fee"
        ):
            continue
        rows.append(
            "| {delay} | {policy} | {coverage:.2%} | {actual:.0f} / {p90:.0f} | {fees} "
            "({bps:.2f} bps) | {mean:+.2%} | {median:+.2%} | {hit:.2%} | {mdd:.2%} | "
            "{insolvent} |".format(
                delay=result["requested_delay_slots"],
                policy=policy_labels[str(result["execution_policy"])],
                coverage=result["coverage_rate_population_weighted"],
                actual=result["actual_delay_slots"]["median"],
                p90=result["actual_delay_slots"]["p90"],
                fees=scenario_labels[str(result["fee_scenario"])],
                bps=result["fee_bps"],
                mean=result["net_mean_return"],
                median=result["net_median_return_unweighted"],
                hit=result["net_hit_rate"],
                mdd=result["max_drawdown_fraction"],
                insolvent="yes" if result["insolvent_under_capital_model"] else "no",
            )
        )
    schema = metrics["schema"]
    behavior = metrics["behavior_parameters"]
    reproduction = metrics.get("reproduction_verification", {})
    table_header = (
        "| Requested delay | Execution | Weighted coverage | Actual delay median / p90 | "
        "Fee scenario | Net mean | Net median | Hit rate | Max drawdown | Insolvent |"
    )
    return f"""# Corrected six-second replica backtest (development validation only)

## Decision

{decision_text} Viability means a strictly positive population-weighted mean, a strictly positive
unweighted median, and a solvent fixed-notional capital path under training-median fees. The
observed viable delay set is `{acceptance["observed_viable_delays"]}`. This is a validation
diagnostic, not an independent profitability estimate, and the final chronological holdout remains
sealed.

The raw deployment-signer balance delta plus strict prior signer-history classifier has
population-adjusted PR-AUC {classifier["population_adjusted_pr_auc"]:.5f}, precision
{classifier["precision"]:.4f}, recall {classifier["recall"]:.4f}, and F1
{classifier["f1"]:.4f} at threshold {classifier["threshold"]:.6f}. It selects
{metrics["selected_sample_rows"]:,} sampled June validation deployments (population weight
{metrics["selected_population_weight"]:,.0f}).

## Frozen behavior and execution

- Wallet events no later than the classifier train end determine the six-second median hold,
  USD {behavior["first_buy_notional_usd"]["median"]:,.2f} median first-buy notional,
  {behavior["fee_bps"]["roundtrip_median"]:.2f} bps median round-trip fees, and
  {behavior["fee_bps"]["roundtrip_p90"]:.2f} bps p90 round-trip fees.
- Entry is the first observed trade at or after deployment slot plus 0/1/2, excluding the
  deployment transaction. `all observed` may fill later than requested; `exact target slot`
  conditions on a trade existing in the target slot.
- Exit is the last observed mark at or before entry time plus six seconds. Post-deployment trades
  are used only for entry/exit marks and returns, never as classifier features.

{table_header}
|---:|:---|---:|:---:|:---|---:|---:|---:|---:|:---:|
{chr(10).join(rows)}

![Corrected fee-sensitive replica returns](figures/replica_validation_backtest.svg)

## Boundaries, integrity, and limitations

- Corrected classification dataset SHA-256: `{metrics["classification_dataset_sha256"]}`;
  {schema["classification_rows"]:,} rows, {schema["classification_unique_tokens"]:,} unique
  tokens, zero strict-history violations, and zero UTC clock mismatches.
- Rebuilt entry-price SHA-256: `{metrics["entry_prices_sha256"]}`;
  {schema["entry_price_rows"]:,} unique token-delay rows and zero corrected-time mismatches.
- Validation ends `{metrics["validation_end_utc"]}`; final holdout starts
  `{metrics["final_holdout_start_utc"]}`; latest selected deployment is
  `{metrics["max_selected_decision_time_utc"]}` and latest outcome mark is
  `{metrics["max_backtest_outcome_utc"]}`.
- Deterministic complete-dictionary reproduction: `{reproduction.get("verified", False)}` over
  {reproduction.get("run_count", 0)} runs. Code parent: `{metrics["code_parent_commit"]}`.
- The entry and exit prices are optimistic observed marks, not guaranteed size-aware fills.
  Population weighting cannot recover path dependence for omitted negatives, and the operating
  threshold was selected on this same validation partition.
"""


def main() -> None:
    metrics = run_validation_backtest()
    reproduced = run_validation_backtest()
    if reproduced != metrics:
        raise AssertionError("deterministic replica backtest rerun did not match")
    metrics["reproduction_verification"] = {
        "verified": True,
        "run_count": 2,
        "comparison": "complete_metrics_dictionary_exact_match",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_report(metrics), encoding="utf-8")
    FIGURE_PATH.write_text(render_backtest_svg(metrics["backtest_results"]), encoding="utf-8")
    append_experiment(
        {
            **metrics,
            "metrics_path": project_relative(METRICS_PATH),
            "metrics_sha256": sha256_file(METRICS_PATH),
            "report_path": project_relative(REPORT_PATH),
            "report_sha256": sha256_file(REPORT_PATH),
            "figure_path": project_relative(FIGURE_PATH),
            "figure_sha256": sha256_file(FIGURE_PATH),
        }
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
