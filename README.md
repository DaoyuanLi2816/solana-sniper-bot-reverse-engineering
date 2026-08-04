# Solana Sniper Bot Reverse-Engineering

Reproducible campaign workspace for the Kaggle community hackathon
[`solana-sniper-bot-reverse-engineering`](https://www.kaggle.com/competitions/solana-sniper-bot-reverse-engineering).

The project has three deliverables:

1. Behavioral analysis of the target wallet.
2. A leak-free, interpretable model of the bot's deployment-time decisions.
3. An honest replica-strategy backtest with delay and fee sensitivity.

## Hard safety boundary

Features used by the classifier and replica strategy must exist at or before
`t_decision`, the token deployment time. Post-deployment trades, candles, and
outcomes are labels or backtest inputs only. They are never decision features.

## Critical experiment erratum

The original sampled-negative pipeline encoded hour in the DuckDB session timezone and weekday
with a Sunday-zero convention, while positives used UTC and Monday-zero. This class-dependent
preprocessing invalidates classification and selection-backtest claims tied to dataset SHA-256
`5fc74ffb...`. The repaired strict-history dataset is
`57af874b0768eaf43c54f04d698cda9e3c3e1d9bf3e1a25c7c69910ecbe8817f`; its time-feature
mismatch count is zero. The frozen raw signer-delta plus strict creator-history model has
standard-validation PR-AUC 0.07028, precision 0.1203, recall 0.2011, and F1 0.1506 at its
training-selected operating point. See [reports/ERRATA.md](reports/ERRATA.md) before interpreting
older reports.

## Frozen development result

The final holdout remains sealed. On strictly pre-holdout outcomes, the optimistic
first-observed-trade proxy passes the predeclared criterion only at requested delay zero. Once the
entry is replaced by the target wallet's training-derived transaction position, the median-fee
replica has mean return +8.11%, median return -5.56%, hit rate 43.53%, maximum drawdown 91.96%, and
crosses zero under the fixed-capital model. That executable position-lag hypothesis is rejected;
the raw zero-slot result is an upper bound, not deployable alpha.

The target wallet itself records mean ROI +10.71%, median ROI +3.05%, hit rate 56.63%, and realized
drawdown 6.89% on 1,372 strictly pre-holdout June buys. These post-deployment outcomes are used only
for description, labeling, and backtesting.

## Layout

- `config/competition.yaml`: live rules and campaign guardrails.
- `data/raw/`: downloaded source files (ignored by Git).
- `data/processed/`: deterministic derived tables (ignored by Git).
- `experiments/manifest.jsonl`: append-only experiment ledger.
- `reports/`: generated analysis, figures, and writeup material.
- `src/solana_sniper/`: reusable data, validation, and modeling code.
- `tests/`: leakage, split, schema, and metric checks.

Competition data is not included in this repository and must not be redistributed. The executed
[public reproduction notebook](notebooks/solana_sniper_reproduction.ipynb) contains code and
aggregate outputs only. Follow [the notebook data-layout instructions](notebooks/README.md) after
obtaining the source files through the official competition channel.

## Initial workflow

```powershell
uv sync
uv run solana-download-wallet
uv run solana-stream-core
uv run solana-audit-wallet
uv run solana-extract-positive
uv run solana-entry-latency
uv run solana-stream-negatives
uv run solana-run-time-integrity
uv run solana-verify-time-remediation
uv run solana-run-baseline
uv run solana-run-boosting
uv run solana-build-creator-history
uv run solana-run-creator-history
uv run solana-run-creator-stability
uv run solana-download-candles
uv run solana-audit-candles
uv run solana-audit-trades-metadata
uv run solana-download-trades
uv run solana-audit-trades
uv run solana-build-entry-prices
uv run solana-run-replica-backtest
uv run solana-run-position-backtest
uv run solana-run-competitor-pnl
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

To rerun the public notebook after staging authorized data privately:

```powershell
uv run --with jupyter --with nbconvert jupyter nbconvert `
  --to notebook --execute --inplace `
  --ExecutePreprocessor.timeout=1800 `
  notebooks/solana_sniper_reproduction.ipynb
```

Dependencies are locked in `uv.lock`. The executed notebook hash and publication state are tracked
in `writeup/submission_manifest.json`; experiment parameters, artifact hashes, code versions, and
negative results remain append-only in `experiments/manifest.jsonl`.

Remote submission is a Kaggle Writeup rather than a leaderboard CSV. A final
submission is allowed only after the public notebook, repository, figures, and
writeup pass the checklist in `config/competition.yaml`.
