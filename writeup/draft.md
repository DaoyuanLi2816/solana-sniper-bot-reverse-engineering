# Reverse-Engineering a Zero-Block Solana Sniper Without Looking Ahead

> Draft. Metrics and conclusions remain provisional until the full negative sample,
> time-based holdout, backtest, and reproduction notebook are complete.

## Executive summary

We study the target wallet as a deployment-time decision system rather than a conventional
price-prediction system. The core question is whether information available at or before token
deployment can explain which launches the wallet buys. All post-deployment activity is isolated
to labels and backtest evaluation.

## 1. Behavioral analysis

### Data provenance

Document the exact source URLs, byte sizes, SHA-256 hashes, row counts, timestamp ranges, and
known limitations from the generated manifests.

### Entry behavior

- Number of tokens bought and total transactions
- Entry-size distribution
- Latency in slots and seconds
- Same-slot transaction-position distribution

### Exit behavior

- Holding-time distribution
- Number and size of partial exits
- Gross and fee-adjusted token-level cash flow
- Burn activity and exceptional paths

## 2. Leak-free feature reverse-engineering

### Decision boundary

Define `t_decision` as token deployment time. Deployer history is truncated strictly before this
time. The current token's deployment transaction may contribute transaction-local information,
but future trades, candles, migration, returns, and realized PnL may not.

### Candidate feature families

1. Deployment transaction structure and transaction position
2. Priority fee, signer balance delta, compute and account footprint
3. Metadata presence, URI family, name and symbol structure
4. Strictly historical deployer behavior and wallet age
5. UTC hour and weekday stability checks

### Validation

Use chronological train, validation, and final holdout periods. Report PR-AUC, precision, recall,
F1, selected operating point, calibration, and stability by month. Accuracy and ROC-AUC are not
primary metrics under the roughly 1:300 class imbalance.

## 3. Replica strategy and backtest

- Define the score and entry threshold before inspecting the final holdout.
- Evaluate 0, 1, and 2 slot execution delays.
- Include priority fee, DEX fee, and conservative slippage assumptions.
- Compare token overlap, PnL, hit rate, and maximum drawdown against the target wallet.
- State operational feasibility separately from historical backtest performance.

## Reproducibility

List the public notebook, repository commit, environment lock, data hashes, commands, and expected
outputs. The notebook must rebuild every reported table and figure end to end.

## Limitations

Preserve negative results, incomplete coverage, external-data limitations, and uncertainty about
live zero-block execution. Do not extrapolate gross historical wallet cash flow into deployable
profit without latency and fee evidence.
