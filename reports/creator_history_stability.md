# Strict prior deployment-signer history stability

## Decision

Retain the strict prior deployment-signer history family. It improves PR-AUC in every predeclared development check after the UTC remediation. The complete metrics dictionary matched across two deterministic runs. This is a
development-period result, not an independent final estimate; the final chronological holdout
remains sealed.

| Fold | Validation period | Without history | With history | Delta | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 2026-04-19 to 2026-05-06 | 0.03078 | 0.06358 | +0.03280 | 0.1103 | 0.1966 | 0.1413 |
| 2 | 2026-05-06 to 2026-05-22 | 0.05780 | 0.08146 | +0.02366 | 0.1335 | 0.2021 | 0.1608 |
| 3 | 2026-05-22 to 2026-06-09 | 0.05522 | 0.07358 | +0.01836 | 0.1379 | 0.2110 | 0.1668 |

### Standard validation operating point

| Metric | Without history | With history | History minus baseline |
|---|---:|---:|---:|
| PR-AUC | 0.05376 | 0.07028 | +0.01652 |
| Precision | 0.0900 | 0.1203 | +0.0303 |
| Recall | 0.2108 | 0.2011 | -0.0097 |
| F1 | 0.1261 | 0.1506 | +0.0244 |
| Threshold | 0.857743 | 0.937099 | n/a |

![Strict prior history stability](figures/creator_history_stability.svg)

## Attribution and interpretable rule

Permutation is evaluated only on standard development validation with the final holdout sealed.
Correlated feature drops are not additive.

| Permuted feature or group | Mean permuted PR-AUC | PR-AUC drop |
|---|---:|---:|
| all four history features | 0.01541 | +0.05487 |
| prior deploy count | 0.04397 | +0.02631 |
| prior active slot count | 0.05611 | +0.01417 |
| observed age seconds | 0.03765 | +0.03263 |
| seconds since previous deploy | 0.03027 | +0.04001 |

| Strict prior feature | Bought median | Sampled not-bought median |
|---|---:|---:|
| creator_prior_deploy_count | 59.000 | 57.000 |
| creator_prior_active_slot_count | 57.000 | 51.000 |
| creator_observed_age_seconds | 7,794,285.000 | 1,975,652.000 |
| creator_seconds_since_previous_deploy | 8,249.000 | 149.000 |

The supported rule is limited to the observed deployment signer or fee payer: the target prefers
signers with longer observed deployment history and more spacing since their previous deploy,
while recurrence counts provide additional context. `tx_signer` is not proven to equal the token
creator address, so these features must not be described as verified wallet age or creator identity.

## `t_decision` boundary and reproducibility

- Dataset: `data/processed/classification_dataset_creator_history.parquet`; SHA-256 `57af874b0768eaf43c54f04d698cda9e3c3e1d9bf3e1a25c7c69910ecbe8817f`.
- Strict-history manifest: `data/processed/creator_history_manifest.json`; SHA-256
  `67ecadcdf40c2e5f7656a6c217a0937b09e14fa1f54e010a398092e88e690570`.
- Rows: 218,350; active rows: 136,412; unique
  tokens: 218,350; invalid strict-history rows:
  0; UTC time mismatches:
  0.
- Same-slot and future deployments are excluded by strict smaller-slot windows. No later trades,
  candles, prices, labels, realized P&L, or future signer history enter these features.
- Baseline uses the raw deployment-signer balance delta; fee-adjusted nonfee outflow remains a
  semantic interpretation proxy, not a demonstrated classifier improvement.
- Maximum evaluated time: `2026-06-09T15:11:49+00:00`.
- Final holdout starts: `2026-06-09T15:12:25+00:00`; no holdout predictions were generated.
- Code parent: `5799329a1a627d26caf4965e0f2a389b32ee0193`.
