# Corrected six-second replica backtest (development validation only)

## Decision

The corrected development result supports the frozen hypothesis: requested delay zero is the only viable delay under the predeclared definition. Viability means a strictly positive population-weighted mean, a strictly positive
unweighted median, and a solvent fixed-notional capital path under training-median fees. The
observed viable delay set is `[0]`. This is a validation
diagnostic, not an independent profitability estimate, and the final chronological holdout remains
sealed.

The raw deployment-signer balance delta plus strict prior signer-history classifier has
population-adjusted PR-AUC 0.07028, precision
0.1203, recall 0.2011, and F1
0.1506 at threshold 0.937099. It selects
335 sampled June validation deployments (population weight
2,183).

## Frozen behavior and execution

- Wallet events no later than the classifier train end determine the six-second median hold,
  USD 201.16 median first-buy notional,
  730.22 bps median round-trip fees, and
  1862.05 bps p90 round-trip fees.
- Entry is the first observed trade at or after deployment slot plus 0/1/2, excluding the
  deployment transaction. `all observed` may fill later than requested; `exact target slot`
  conditions on a trade existing in the target slot.
- Exit is the last observed mark at or before entry time plus six seconds. Post-deployment trades
  are used only for entry/exit marks and returns, never as classifier features.

| Requested delay | Execution | Weighted coverage | Actual delay median / p90 | Fee scenario | Net mean | Net median | Hit rate | Max drawdown | Insolvent |
|---:|:---|---:|:---:|:---|---:|---:|---:|---:|:---:|
| 0 | all observed | 100.00% | 0 / 1 | gross (0 bps) (0.00 bps) | +82.16% | +23.39% | 59.23% | 2.34% | no |
| 0 | all observed | 100.00% | 0 / 1 | training median (730.22 bps) | +74.86% | +16.08% | 49.43% | 3.20% | no |
| 0 | all observed | 100.00% | 0 / 1 | training p90 (1862.05 bps) | +63.54% | +4.77% | 40.27% | 5.31% | no |
| 0 | exact target slot | 76.23% | 0 / 0 | training median (730.22 bps) | +98.52% | +19.82% | 58.23% | 2.81% | no |
| 1 | all observed | 100.00% | 1 / 2 | gross (0 bps) (0.00 bps) | +2.99% | -2.95% | 39.67% | 46.84% | no |
| 1 | all observed | 100.00% | 1 / 2 | training median (730.22 bps) | -4.32% | -10.25% | 30.92% | 233.08% | yes |
| 1 | all observed | 100.00% | 1 / 2 | training p90 (1862.05 bps) | -15.63% | -21.57% | 21.67% | 692.15% | yes |
| 1 | exact target slot | 66.51% | 1 / 1 | training median (730.22 bps) | -3.16% | -10.85% | 37.81% | 141.88% | yes |
| 2 | all observed | 100.00% | 2 / 4 | gross (0 bps) (0.00 bps) | +40.16% | -1.17% | 41.91% | 3.50% | no |
| 2 | all observed | 100.00% | 2 / 4 | training median (730.22 bps) | +32.86% | -8.47% | 28.13% | 8.29% | no |
| 2 | all observed | 100.00% | 2 / 4 | training p90 (1862.05 bps) | +21.54% | -19.79% | 22.86% | 24.94% | yes |
| 2 | exact target slot | 45.03% | 2 / 2 | training median (730.22 bps) | -5.07% | -9.58% | 33.98% | 169.59% | yes |

![Corrected fee-sensitive replica returns](figures/replica_validation_backtest.svg)

## Boundaries, integrity, and limitations

- Corrected classification dataset SHA-256: `57af874b0768eaf43c54f04d698cda9e3c3e1d9bf3e1a25c7c69910ecbe8817f`;
  218,350 rows, 218,350 unique
  tokens, zero strict-history violations, and zero UTC clock mismatches.
- Rebuilt entry-price SHA-256: `8971bc3c210d6d8a7b98b21fd6a0a66eefcc2aaf5b4e8b2fd27c6221d9cdad83`;
  114,333 unique token-delay rows and zero corrected-time mismatches.
- Validation ends `2026-06-09T15:11:49+00:00`; final holdout starts
  `2026-06-09T15:12:25+00:00`; latest selected deployment is
  `2026-06-09T15:07:15+00:00` and latest outcome mark is
  `2026-06-09T15:07:22+00:00`.
- Deterministic complete-dictionary reproduction: `True` over
  2 runs. Code parent: `94335c2c671940dc2b519096e97a38ba68b9768e`.
- The entry and exit prices are optimistic observed marks, not guaranteed size-aware fills.
  Population weighting cannot recover path dependence for omitted negatives, and the operating
  threshold was selected on this same validation partition.
