# Zero-slot transaction-position feasibility (validation only)

## Hypothesis

The previous zero-slot backtest entered at the first observed trade after deployment. This run
tests whether its profitability survives a position that resembles the target wallet's observed
same-slot timing. The only changed assumption is entry position; the classifier, operating point,
six-second exit mark, notional, fee scenarios, population weights, and chronological boundaries
are unchanged.

The position lag is frozen from target-wallet observations no later than the classifier training
cutoff (`2026-05-18 16:44:18 UTC`). Among 7,377 training-period same-slot buys, the wallet entered
a median 112 transaction positions after deployment (p10 25, p25 48, p75 224, p90 421). Only
1.76% arrived within ten transaction positions.

By contrast, among the 344 selected validation tokens for which the previous proxy found a trade
in the deployment slot, the first post-deployment trade had a median position gap of one. Only
22.97% of those first trades were at least 112 positions after deployment. The prior price was
therefore usually much earlier than the target wallet's own observed position.

## Frozen policy

- Selection: the unchanged creator-history model at threshold 0.930016.
- Classification validation: PR-AUC 0.08934, precision 0.12735, recall 0.23493, F1 0.16517.
- Entry: first trade in the deployment slot whose transaction index is at least deployment index
  plus 112.
- Exit: last observed trade mark at or before entry plus six seconds.
- Notional: training-wallet median first buy, USD 201.16.
- Fees: gross, training median 730.22 bps, and training p90 1,862.05 bps.
- Evaluation: June portion of validation only; the final chronological holdout remains sealed.

The policy attempts 387 sampled candidates (population weight 2,523) and fills 275 (population
weight 1,451): 71.06% sampled coverage and 57.51% population-weighted coverage. Filled positions
have median transaction gap 219 and p90 559.2.

## Results

| fee scenario | net weighted mean | net unweighted median | weighted hit rate | max drawdown | insolvent |
|:---|---:|---:|---:|---:|:---:|
| Gross | +12.81% | +0.52% | 46.04% | 47.95% | no |
| Training median | +5.51% | -6.78% | 38.18% | 111.97% | yes |
| Training p90 | -5.81% | -18.10% | 31.36% | 258.96% | yes |

The prior first-post-deployment proxy reported weighted mean returns of +40.38%, +33.08%, and
+21.76% in the same three fee scenarios. Applying the frozen transaction-position lag reduces
each mean by 27.56 percentage points. At median fees, the mean remains positive only because of a
right tail (gross p90 +78.50%, p99 +236.30%); the median trade loses money and the tight capital
model crosses zero. At p90 fees, both mean and median are negative.

![Position-lag validation comparison](figures/position_lag_validation_backtest.svg)

## Decision

The hypothesis is **rejected**: zero-slot profitability is not robust to the target wallet's
training-derived transaction position. The earlier first-trade result is retained as an optimistic
price upper bound, not a feasible candidate backtest. The final holdout will not be opened for
this policy.

Transaction index is still only a position proxy; it does not prove that a new replica can observe
deployment and react in the same slot. The exit remains a mark rather than an executable sell,
and population weighting cannot reconstruct the exact path of omitted negatives. These limitations
all lean against claiming executable profitability.

The reproducible aggregate is
`reports/generated/position_lag_validation_backtest.json` (SHA-256
`319f88665a85b02e89505d74d98813224f848154f590c0cbf8fbc6fbae2a813d`). The figure SHA-256 is
`eb7b637c01fd9ed4207d29dc777a585b682d27948a253919ee09afa2f51ebd24`.
