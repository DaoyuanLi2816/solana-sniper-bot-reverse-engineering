# Leak-free temporal feature attribution

## Decision

The hypothesis is narrowly supported on development folds: strict prior creator history ranks first and has a positive permutation drop in every fold. It is not an overwhelming lead: the mean drop is 0.06825 versus 0.06340 for transaction structure. This is an interpretability result, not a new performance claim. The final
chronological holdout remains sealed.

| Group | Mean temporal PR-AUC drop | Minimum fold drop | Positive folds | Standard validation drop |
|---|---:|---:|---:|---:|
| creator history | 0.06825 | 0.06146 | 3/3 | 0.07201 |
| transaction structure | 0.06340 | 0.05671 | 3/3 | 0.06748 |
| decision time | 0.01866 | 0.01239 | 3/3 | 0.01858 |
| metadata | 0.01242 | 0.00678 | 3/3 | 0.00326 |
| fee and compute | 0.00642 | 0.00339 | 3/3 | 0.01899 |
| transaction error | 0.00000 | 0.00000 | 0/3 | 0.00000 |

## Temporal checks

| Fold | Validation period | Baseline PR-AUC | Creator-history drop | Largest group |
|---|---|---:|---:|---|
| 1 | 2026-04-19 to 2026-05-06 | 0.07871 | 0.06146 | creator history |
| 2 | 2026-05-06 to 2026-05-22 | 0.09951 | 0.08095 | creator history |
| 3 | 2026-05-22 to 2026-06-09 | 0.08035 | 0.06233 | creator history |

## Top-10 individual features

The rank is the mean weighted PR-AUC loss across three expanding chronological validation folds.
The sampled class medians are a directional description only; correlated importances are not
additive.

| Rank | Feature | Mean temporal drop | Positive folds | Bought median | Not-bought median |
|---:|---|---:|---:|---:|---:|
| 1 | Signer lamport delta | 0.05309 | 3/3 | -2,846,828,720 | -266,968,190 |
| 2 | Seconds since prior deploy | 0.04657 | 3/3 | 8,249.0 | 149.0 |
| 3 | Observed creator age | 0.03902 | 3/3 | 7,794,285 | 1,975,652 |
| 4 | Prior deploy count | 0.02184 | 3/3 | 59.000 | 57.000 |
| 5 | Decision hour UTC | 0.02093 | 3/3 | 13.000 | 12.000 |
| 6 | Inner instruction count | 0.01477 | 3/3 | 26.000 | 28.000 |
| 7 | Post-token balance count | 0.01123 | 3/3 | 2.000 | 3.000 |
| 8 | Metadata URI length | 0.00700 | 3/3 | 64.000 | 67.000 |
| 9 | Transaction fee | 0.00592 | 3/3 | 96,941.0 | 33,687.0 |
| 10 | Prior active-slot count | 0.00481 | 2/3 | 57.000 | 51.000 |

![Temporal permutation importance](figures/feature_attribution.svg)

## Plain-language rule hypothesis

The model is most consistent with two strong, complementary screens: creator maturity/spacing and
the construction of the deployment transaction. It does **not** establish that either screen is
evaluated first. The four creator-history associations are:

- **Prior deploy count:** selected tokens have a higher validation median (59.000 vs 57.000); temporal mean PR-AUC drop is 0.02184.
- **Prior active-slot count:** selected tokens have a higher validation median (57.000 vs 51.000); temporal mean PR-AUC drop is 0.00481.
- **Observed creator age:** selected tokens have a higher validation median (7,794,285 vs 1,975,652); temporal mean PR-AUC drop is 0.03902.
- **Seconds since prior deploy:** selected tokens have a higher validation median (8,249.0 vs 149.0); temporal mean PR-AUC drop is 0.04657.

These are associations learned from strictly pre-decision fields. They do not prove the wallet's
implementation or a causal trading rule. The strongest individual diagnostic is signer lamport
delta, and the transaction-structure family is close to creator history, so the practical rule
hypothesis must retain both. Prior active-slot count is also unstable (positive in only two of three
folds), despite appearing in the top ten.

## Reproducibility and holdout boundary

- Dataset: `data/processed/classification_dataset_creator_history.parquet`; SHA-256 `5fc74ffb4d7ac5a0fd26ff5a4cb4326a89d227d273fd3c583ea88d827a018f12`.
- Rows: 218,350; unique tokens: 218,350;
  positives: 15,927;
  negatives: 202,423.
- Temporal permutations: 3 deterministic repeats per feature/group;
  standard validation: 5 repeats.
- Maximum evaluated time: `2026-06-09T15:11:49+00:00`.
- Final holdout starts: `2026-06-09T15:12:25+00:00` and was not predicted.
- Code parent: `1e987d909ce515fc2a0183fa791222fc2548aaf0`.
