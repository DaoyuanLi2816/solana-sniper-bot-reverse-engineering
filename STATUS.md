# Campaign status

Last verified: 2026-08-01 08:15 UTC

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
- Started a deterministic 1-in-25 sample across all 5,059,880 negative deployments.
- Created Codex heartbeat `solana-sniper-bot`, every three hours through the deadline.

## First behavioral findings

- Median first entry size: approximately USD 184.
- Median observed holding time: 6 seconds.
- Median sell transactions per token: 4.
- Same-slot entry share: approximately 79.6%.

These are descriptive findings, not evidence that a classifier or replica strategy is competitive.

## Current blocker

The first classification baseline waits for the full time-spanning negative sample and index join.
Do not substitute an early-window-only negative sample because absolute time would become a trivial,
non-generalizing separator.
