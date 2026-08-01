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

### Provisional deployment-transaction baseline

We sampled every 25th not-bought deployment across the complete negative archive, then restored
the population ratio through a 25x negative evaluation weight. A strict final time holdout has a
0.4597% weighted prevalence. Logistic regression reaches 0.01089 PR-AUC, while changing only the
model family to histogram gradient boosting reaches 0.04547 PR-AUC (9.89x prevalence). At the
validation-selected threshold, the nonlinear model records 8.24% precision and 14.3% recall on
the final period. These metrics establish selection signal; they do not establish trading profit.

Adding only strictly prior-slot deployment history for the deployment signer raises validation
PR-AUC from 0.06197 to 0.08934 (+44.2%); the final test remains sealed. Bought tokens are less often
from first-seen signers (9.61% versus 20.58%), while their signers have a longer observed history
(90.1 versus 22.5 median days) and are less likely to have deployed again within minutes (2.36
hours versus 2.47 minutes since the prior deployment). The raw prior-deployment-count medians are
similar (55 bought versus 56 not bought), suggesting cadence and wallet maturity matter more than
simple deploy volume.

The result is temporally stable rather than confined to one split. In three expanding validation
windows, adding creator history changes population-adjusted PR-AUC by +0.02976, +0.02792, and
+0.01869. The shrinking late-window gain is a limitation, but all three deltas remain positive.
Jointly permuting the creator feature family reduces standard-validation PR-AUC from 0.08934 to
0.01735. Recency and observed wallet age cause the largest individual drops; correlated-feature
permutation effects are not additive. All of these analyses end before the untouched final-test
boundary.

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
