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
