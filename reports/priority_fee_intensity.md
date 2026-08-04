# Priority-fee intensity at deployment

## Decision

The intensity replacement is rejected because at least one development check exceeds the predeclared 0.002 PR-AUC loss margin. This is a two-run-reproduced post-UTC-remediation development result, not an
independent final estimate; the final chronological holdout remains sealed.

| Fold | Validation period | Absolute PR-AUC | Intensity PR-AUC | Delta | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 2026-04-19 to 2026-05-06 | 0.06358 | 0.06001 | -0.00357 | 0.1130 | 0.1966 | 0.1435 |
| 2 | 2026-05-06 to 2026-05-22 | 0.08146 | 0.08234 | +0.00088 | 0.1216 | 0.2545 | 0.1646 |
| 3 | 2026-05-22 to 2026-06-09 | 0.07358 | 0.07129 | -0.00228 | 0.1476 | 0.1709 | 0.1584 |

### Standard validation operating point

| Metric | Absolute priority fee | Fee per compute unit | Intensity minus absolute |
|---|---:|---:|---:|
| PR-AUC | 0.07028 | 0.06949 | -0.00079 |
| Precision | 0.1203 | 0.1146 | -0.0058 |
| Recall | 0.2011 | 0.2188 | +0.0176 |
| F1 | 0.1506 | 0.1504 | -0.0002 |
| Threshold | 0.937099 | 0.928165 | n/a |

![Absolute priority fee versus fee intensity](figures/priority_fee_intensity.svg)

## Interpretation at `t_decision`

The feature is `priority_fee_lamports / compute_units`. Both values come from the observed
deployment transaction. It is therefore available at `t_decision`; no later trade, price, candle,
label, realized P&L, or future deployer history enters the model.

On standard validation, bought deployments have median intensity 0.4055 lamports/CU
(p90 5.1911), versus 0.1050 (p90
5.2277) for sampled not-bought deployments. The raw-to-intensity Spearman
correlation is 0.991586. Consumed compute units are not the
requested compute limit, so this is a realized transaction-urgency proxy rather than an exact bid.

## Reproducibility boundary

- Dataset: `data/processed/classification_dataset_creator_history.parquet`; SHA-256 `57af874b0768eaf43c54f04d698cda9e3c3e1d9bf3e1a25c7c69910ecbe8817f`.
- Rows: 218,350; active rows: 136,412; unique
  tokens: 218,350; nonpositive compute rows:
  0.
- Exactly one feature is replaced on the current raw signer-delta plus strict-history baseline;
  the fee-adjusted outflow proxy is not used.
- Strict-history violations and UTC hour/weekday mismatches are all zero.
- Two complete deterministic runs matched the metrics dictionary exactly.
- Maximum evaluated time: `2026-06-09T15:11:49+00:00`.
- Final holdout starts: `2026-06-09T15:12:25+00:00`; no holdout predictions were generated.
