# Campaign status

Last verified: 2026-08-01 11:29 UTC

## Remote state

- Competition: `solana-sniper-bot-reverse-engineering`
- Account entered: yes (verified with authenticated Kaggle CLI)
- Deadline: 2026-08-14 21:00 UTC
- Prize advertised on Overview: one first-place award of USD 1,000, paid in USDC
- Submission type: Kaggle Writeup with public notebook and repository
- Authenticated API team count: 4; Writeups page lists four hidden submitted projects
- Our team: `DL`, one member, no Writeup created or submitted yet
- International remote participation: eligible on equal terms (host confirmation)
- LLM and automated-ML tools: allowed when reasonably accessible and minimally costly

## Rules risk requiring monitoring

The published prize text is internally inconsistent. Overview advertises a single USD 1,000
first-place award, while the Rules page still contains unfilled sponsor/prize placeholders and a
USD 600/300/100 split. No existing discussion resolves this discrepancy. Treat USD 1,000 as the
advertised total rather than a guaranteed single payout until the host corrects or clarifies the
Rules page. The Team UI currently says ten members while the competition-specific Rules say five;
use the stricter five-person limit. The competition data must not be redistributed outside
eligible participants.

## Completed locally

- Downloaded and SHA-256 verified the three target-wallet files.
- Stream-extracted the three positive-class core files without storing the 41.5 GB tar.
- Audited 87,007 target-wallet activities and 16,163 bought tokens.
- Extracted 15,927 positive deployment-time feature rows.
- Matched every positive deployment to its first wallet buy.
- Completed a deterministic 1-in-25 sample across all 5,059,880 negative deployments.
- Built 202,423 negative deployment-token rows from 202,395 sampled transactions and a
  218,350-row classification table. The difference is 28 sampled transactions that deployed two
  tokens; token addresses remain unique.
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

## Creator-history hypothesis (validation only)

Adding four strictly prior-slot features from the deployment signer improved population-adjusted
validation PR-AUC from 0.06197 to 0.08934 (+44.2%). At the validation-selected threshold, precision
is 12.74%, recall 23.49%, and F1 0.165. The final test partition remains withheld until a candidate
is frozen.

During the validation window, bought tokens were less often a signer's first observed deployment
(9.61% versus 20.58% for sampled non-buys). Their signers had a much longer median observed age
(90.1 versus 22.5 days) and a longer median interval since the previous deployment (2.36 hours
versus 2.47 minutes). This supports an interpretable preference for established, less immediately
spammy deployers rather than raw deployment count alone.

The nonlinear result is promising evidence of structured deployment-time selection, but it is not
yet a profitable replica strategy. The next priority is creator-history attribution/stability,
followed by fee/slippage/0-1-2-slot backtesting. The first unweighted logistic run is preserved in
the experiment ledger with an erratum rather than silently deleted.
