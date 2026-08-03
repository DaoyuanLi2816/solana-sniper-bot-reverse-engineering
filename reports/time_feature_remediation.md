# UTC time-feature remediation verification

The cached negative sample and local deployment index were rebuilt without a full raw-block scan.
Every repaired row now uses UTC hour and Monday-zero weekday derived from `decision_time`.

- Decision: **verified canonical UTC rebuild**.
- Repaired dataset: `data/processed/classification_dataset_creator_history.parquet`; SHA-256
  `57af874b0768eaf43c54f04d698cda9e3c3e1d9bf3e1a25c7c69910ecbe8817f`.
- Rows: 218,350; unique tokens:
  218,350; hour mismatches:
  0; weekday mismatches:
  0.
- The three fold metrics and standard metric exactly match the deterministically ordered
  pre-fix canonical snapshot.
- Corrected standard validation: PR-AUC
  0.06980, precision 0.1383,
  recall 0.1747, F1 0.1544, threshold
  0.945563.
- Maximum evaluated time: `2026-06-09T15:11:49+00:00`; final holdout starts
  `2026-06-09T15:12:25+00:00` and remains sealed.

The earlier 0.09933 standard PR-AUC is invalid because it included class-dependent time
preprocessing. The corrected 0.06980 value is the current evidence-backed baseline.
