# Experiment errata

## 2026-08-03 — class-dependent time-feature construction

The pre-fix strict-history dataset
`5fc74ffb4d7ac5a0fd26ff5a4cb4326a89d227d273fd3c583ea88d827a018f12` contains a
class-dependent preprocessing defect:

- All 202,423 sampled negative rows stored a session-local hour rather than UTC hour.
- 150,364 negative rows stored a Sunday-zero weekday rather than pandas Monday-zero weekday.
- All 15,927 positive rows used the intended UTC hour and Monday-zero weekday.

Consequently, every classification metric, attribution claim, selector comparison, and
selection-dependent backtest tied to that dataset hash is superseded. This includes the old
creator-history stability/attribution, fee-adjusted outflow selector, priority-fee, metadata,
creator-rate, creator-slot-batching, and outflow replica/position reports. Their files remain as
negative and historical evidence; they must not be cited as current model quality or strategy
performance.

The target-wallet behavioral and realized-P&L summaries derived directly from trade events do not
use these classifier time features, but their split-boundary hashes should still be rechecked
before final submission.

## Verified remediation

- Negative raw feature scan reused: yes; no full archive rescan or 429 GiB block download.
- Repaired sampled-negative features:
  `d49aae26fe82313d6570d22a68d21ee1f14422a0f41557c9ea7fc1754b2a0446`.
- Repaired classification dataset:
  `9c8f738b209c01328122f431da21450d33f58f1931b8986b133da9b3d2c8958b`.
- Repaired strict-history dataset:
  `57af874b0768eaf43c54f04d698cda9e3c3e1d9bf3e1a25c7c69910ecbe8817f`.
- Rows / unique tokens: 218,350 / 218,350.
- UTC-hour mismatches / weekday mismatches: 0 / 0.
- Corrected standard validation: PR-AUC 0.06980, precision 0.1383, recall 0.1747,
  F1 0.1544, operating threshold 0.945563.
- Final holdout remains sealed; maximum development evaluation time is
  `2026-06-09T15:11:49+00:00`, before the holdout start
  `2026-06-09T15:12:25+00:00`.

Detailed evidence is in [time_feature_integrity.md](time_feature_integrity.md) and
[time_feature_remediation.md](time_feature_remediation.md).

## Revalidated selector after remediation

The fee-adjusted deployment-signer outflow proxy was rerun on repaired strict-history dataset
`57af874b0768eaf43c54f04d698cda9e3c3e1d9bf3e1a25c7c69910ecbe8817f`. Its PR-AUC deltas
versus the raw signer balance delta were -0.00194, -0.00106, and +0.00217 in the three expanding
folds, and -0.00048 in standard validation. It remains within the predeclared 0.002
noninferiority margin in every development check, but the minimum delta is only 0.000058 above
that boundary. It is retained for semantic interpretation, not claimed as a performance
improvement. The corrected metrics and operating point are in
[deployment_outflow_proxy.md](deployment_outflow_proxy.md); the final holdout remains sealed.

## Revalidated strict history after remediation

The strict prior deployment-signer history family was rerun on repaired strict-history dataset
`57af874b0768eaf43c54f04d698cda9e3c3e1d9bf3e1a25c7c69910ecbe8817f`. Its PR-AUC deltas
versus an otherwise identical model without the four history fields were +0.03280, +0.02366, and
+0.01836 in the three expanding folds, and +0.01652 in standard validation. It therefore passes
the predeclared requirement that every development check improve. The corrected standard model
has PR-AUC 0.07028, precision 0.1203, recall 0.2011, and F1 0.1506 at threshold 0.937099. This
supports strict prior signer recurrence, observed maturity, and spacing as the current core
explainable signal; it does not prove creator identity or true on-chain wallet age. Detailed
evidence is in [creator_history_stability.md](creator_history_stability.md); the final holdout
remains sealed.

## Revalidated feature attribution after remediation

The complete grouped and individual permutation analysis was rerun on repaired strict-history
dataset `57af874b0768eaf43c54f04d698cda9e3c3e1d9bf3e1a25c7c69910ecbe8817f`. Strict prior
deployment-signer history ranks first in every expanding fold, with mean temporal PR-AUC drop
0.05661, narrowly ahead of transaction structure at 0.05262. All four strict history features have
positive drops in all three folds. The corrected Top-10 begins with raw signer lamport delta,
seconds since prior deploy, observed signer deployment age, and prior signer deploy count. The
decision-time group falls to mean drop -0.00029, so the old apparent time-of-day importance must
not be cited. The standard development operating point remains PR-AUC 0.07028, precision 0.1203,
recall 0.2011, and F1 0.1506 at threshold 0.937099. Detailed current evidence is in
[feature_attribution.md](feature_attribution.md); the final holdout remains sealed.

## Revalidated replica backtest after remediation

The raw deployment-signer delta plus strict prior signer-history selector was rerun twice with an
exact complete-dictionary match on repaired dataset
`57af874b0768eaf43c54f04d698cda9e3c3e1d9bf3e1a25c7c69910ecbe8817f`. The June entry-price
audit now references that corrected source; its 114,333 token-delay rows retain SHA-256
`8971bc3c210d6d8a7b98b21fd6a0a66eefcc2aaf5b4e8b2fd27c6221d9cdad83`, confirming the UTC
repair did not alter deployment context or post-deployment price ordering.

At the training-derived median round-trip fee of 730.22 bps under all-observed execution, the
requested 0/1/2-slot results are respectively: net weighted mean +74.86% / -4.32% / +32.86%,
unweighted median +16.08% / -10.25% / -8.47%, hit rate 49.43% / 30.92% / 28.13%, and maximum
drawdown 3.20% / 233.08% / 8.29%. Delay 1 crosses zero; delay 2 remains solvent but fails the
positive-median requirement, so only requested delay zero passes the predeclared viability
definition. The positive delay-2 mean is tail-driven and must not be presented as typical-trade
profitability. These are validation diagnostics based on optimistic observed price marks, not
guaranteed fills or an independent estimate. Detailed evidence is in
[replica_validation_backtest.md](replica_validation_backtest.md); the final holdout remains sealed.

## Revalidated priority-fee intensity after remediation

Replacing absolute deployment priority fee with priority fee per consumed compute unit was rerun
twice on the repaired raw signer-delta plus strict-history baseline. The complete result dictionary
matched across runs. Intensity-minus-absolute PR-AUC was -0.00357, +0.00088, and -0.00228 in the
three expanding folds, so folds 1 and 3 exceed the predeclared 0.002 loss margin. Standard
validation also fell from PR-AUC 0.07028 to 0.06949; the intensity operating point has precision
0.1146, recall 0.2188, and F1 0.1504 at threshold 0.928165. The intensity replacement is rejected,
and absolute priority fee remains in the current selector. Fee per consumed compute unit may be
reported descriptively as a realized deployment-transaction urgency proxy, but consumed compute
is not the requested compute limit and the proxy must not be claimed as a classifier improvement.
Detailed evidence is in [priority_fee_intensity.md](priority_fee_intensity.md); the final holdout
remains sealed.

## Revalidated transaction-position and competitor comparison after remediation

The executable same-slot transaction-position diagnostic and target-wallet comparison were rerun
twice with exact complete-dictionary matches on repaired strict-history dataset
`57af874b0768eaf43c54f04d698cda9e3c3e1d9bf3e1a25c7c69910ecbe8817f`. Entry-latency and
entry-price decision times have zero token-level mismatch with the corrected classification table;
UTC-hour, weekday, and strict-history violation counts are also zero.

The training-only wallet median position remains 112 transactions after deployment. The corrected
selector attempts 335 validation candidates and obtains 237 same-slot fills, for 57.03%
population-weighted coverage. At the training-derived median round-trip fee, weighted mean return
is +8.11%, but the unweighted median is -5.56%, hit rate is 43.53%, maximum drawdown is 91.96%,
and the tight fixed-notional capital path briefly crosses zero. The position-lag candidate is
therefore rejected under the predeclared requirement of positive mean, positive median, and
solvency. The first-observed-trade zero-slot result remains an optimistic price proxy, not an
executable claim.

The target wallet remains profitable after its recorded gas and DEX fees in the same strictly
pre-holdout window: mean ROI +10.71%, median ROI +3.05%, hit rate 56.63%, and realized drawdown
6.89% across 1,372 actual buys. This is descriptive competitor evidence and does not transfer to
the replica because entries, sizing, sampling, and execution differ. Detailed evidence is in
[position_lag_validation_backtest.md](position_lag_validation_backtest.md) and
[competitor_fee_pnl.md](competitor_fee_pnl.md); the final holdout remains sealed.
