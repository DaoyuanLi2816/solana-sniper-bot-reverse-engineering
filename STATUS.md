# Campaign status

Last verified: 2026-08-01 08:39 UTC

## Remote state

- Competition: `solana-sniper-bot-reverse-engineering`
- Account entered: yes (verified with authenticated Kaggle CLI)
- Deadline: 2026-08-14 21:00 UTC
- Prize advertised on Overview: one first-place award of USD 1,000, paid in USDC
- Submission type: Kaggle Writeup with public notebook and repository
- Current teams at verification: 4
- International remote participation: eligible on equal terms (host confirmation)
- LLM and automated-ML tools: allowed when reasonably accessible and minimally costly

## Rules risk requiring monitoring

The published prize text is internally inconsistent. Overview advertises a single USD 1,000
first-place award, while the Rules page still contains unfilled sponsor/prize placeholders and a
USD 600/300/100 split. No existing discussion resolves this discrepancy. Treat USD 1,000 as the
advertised total rather than a guaranteed single payout until the host corrects or clarifies the
Rules page. The competition data must not be redistributed outside eligible participants.

## Completed locally

- Downloaded and SHA-256 verified the three target-wallet files.
- Stream-extracted the three positive-class core files without storing the 41.5 GB tar.
- Audited 87,007 target-wallet activities and 16,163 bought tokens.
- Extracted 15,927 positive deployment-time feature rows.
- Matched every positive deployment to its first wallet buy.
- Completed a deterministic 1-in-25 sample across all 5,059,880 negative deployments.
- Built 202,395 negative deployment-time rows and a 218,322-row classification table.
- Ran strict chronological train/validation/final-test comparisons with population weighting.
- Created Codex heartbeat `solana-sniper-bot`, every three hours through the deadline.

## First behavioral findings

- Median first entry size: approximately USD 184.
- Median observed holding time: 6 seconds.
- Median sell transactions per token: 4.
- Same-slot entry share: approximately 79.6%.

These are descriptive findings, not evidence that a classifier or replica strategy is competitive.

## First model results

All reported precision and PR-AUC values below weight each sampled negative as 25 deployments.
The operating threshold was selected on validation and evaluated once on the later final test.

- Population prevalence in the final test: 0.4597%.
- Logistic baseline final PR-AUC: 0.01089 (2.37x prevalence); precision 1.66%, recall 16.0%.
- Histogram gradient boosting final PR-AUC: 0.04547 (9.89x prevalence); precision 8.24%,
  recall 14.3%.

The nonlinear result is promising evidence of structured deployment-time selection, but it is not
yet a profitable replica strategy. The next priority is strictly historical deployer features,
followed by feature attribution and a fee/slippage/0-1-2-slot backtest. The first unweighted
logistic run is preserved in the experiment ledger with an erratum rather than silently deleted.
