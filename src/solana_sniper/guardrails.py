from collections.abc import Iterable

POST_DECISION_MARKERS = {
    "future_price",
    "future_return",
    "mcap_after",
    "post_deploy",
    "realized_pnl",
    "sell_price",
    "trade_after_deploy",
}


def assert_feature_names_are_pre_decision(feature_names: Iterable[str]) -> None:
    """Reject feature names that explicitly encode post-decision information."""
    violations = []
    for name in feature_names:
        normalized = name.lower()
        if any(marker in normalized for marker in POST_DECISION_MARKERS):
            violations.append(name)
    if violations:
        raise ValueError(f"Post-decision feature names are forbidden: {sorted(violations)}")


def assert_history_precedes_decision(history_times, decision_times) -> None:
    """Require every historical event timestamp to be no later than its decision time."""
    bad = history_times > decision_times
    if bool(bad.any()):
        raise ValueError(f"Found {int(bad.sum())} future-history rows")
