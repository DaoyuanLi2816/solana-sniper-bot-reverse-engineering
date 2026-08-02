# Deployment-signer fee-adjusted outflow proxy

## Decision

The fee-adjusted deployment-signer outflow proxy is retained for interpretation. It stayed within the predeclared 0.002 PR-AUC noninferiority margin in every development check. It also increased PR-AUC in all three expanding folds and standard validation, and the
full metrics dictionary matched exactly across two deterministic runs. This is a reproduced
**development-period** improvement, not an independent final estimate; the final chronological
holdout remains sealed.

| Fold | Validation period | Raw PR-AUC | Proxy PR-AUC | Delta | Precision | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 2026-04-19 to 2026-05-06 | 0.07871 | 0.08052 | +0.00181 | 0.1553 | 0.1874 | 0.1699 |
| 2 | 2026-05-06 to 2026-05-22 | 0.09951 | 0.10094 | +0.00143 | 0.1579 | 0.2442 | 0.1918 |
| 3 | 2026-05-22 to 2026-06-09 | 0.08035 | 0.08714 | +0.00680 | 0.1421 | 0.1939 | 0.1640 |

### Standard validation operating point

| Metric | Raw balance delta | Fee-adjusted outflow | Proxy minus raw |
|---|---:|---:|---:|
| PR-AUC | 0.08934 | 0.09933 | +0.00999 |
| Precision | 0.1274 | 0.1736 | +0.0463 |
| Recall | 0.2349 | 0.1714 | -0.0635 |
| F1 | 0.1652 | 0.1725 | +0.0073 |
| Selected threshold | 0.930016 | 0.948837 | n/a |

![Raw balance delta versus fee-adjusted outflow](figures/deployment_outflow_proxy.svg)

## Exact meaning at `t_decision`

The proxy is `max(-signer_lamport_delta - tx_fee_lamports, 0)`. Both inputs come from the
deployment transaction's balance and fee metadata, so no later trade, candle, outcome, or future
history enters the feature.

The precise name is **deployment-signer/fee-payer net nonfee outflow**, not creator dev-buy.
`creator_address` is present in only 0.49% of rows
(1,078/218,350); among those rows, its equality
rate with `tx_signer` is 0.00%.
Without decoding every program transfer, the net outflow may mix buy principal, account rent,
account funding, and protocol transfers.

## Interpretable association

On standard validation, bought tokens have median signer nonfee outflow of
2.847 SOL (p10 0.024, p90 7.011), versus
0.267 SOL for sampled not-bought tokens (p10
0.020, p90 3.797). The raw-to-proxy Spearman
correlation is -0.998638.

The defensible rule hypothesis is therefore: **the target favors deployment transactions whose
signer commits substantially more nonfee lamports**, alongside the previously established signer
history signal. Calling this amount a pure dev-buy would exceed the evidence.

## Reproducibility and boundary

- Dataset: `data/processed/classification_dataset_creator_history.parquet`; SHA-256 `5fc74ffb4d7ac5a0fd26ff5a4cb4326a89d227d273fd3c583ea88d827a018f12`.
- Rows: 218,350; unique tokens: 218,350;
  duplicate transaction-hash rows: 28.
- Model: the frozen HGB parameters, with exactly one feature replaced and feature count unchanged.
- Reproduction: 2 deterministic runs matched
  the complete metrics dictionary exactly.
- Maximum evaluated time: `2026-06-09T15:11:49+00:00`.
- Final holdout starts: `2026-06-09T15:12:25+00:00`; no holdout predictions were generated.
- Code parent: `955e4bcc9b221263e2082bcac42185235ad5578d`.
