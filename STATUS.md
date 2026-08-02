# Campaign status

Last verified: 2026-08-02 05:42 UTC

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

## Creator-history stability and attribution

The same model and hyperparameters were compared with and without creator history in three
expanding chronological folds ending before the final holdout. PR-AUC deltas were +0.02976,
+0.02792, and +0.01869: positive in 3/3 folds, with some late-period decay. The mean absolute
delta was +0.02546.

On the standard validation window, jointly permuting all four creator features reduced PR-AUC from
0.08934 to 0.01735. The largest individual drops came from seconds since the previous deployment
(0.05131) and observed creator age (0.04805), followed by prior deployment count (0.03487) and
prior active-slot count (0.02047). Individual drops are not additive because these features are
correlated.

No prediction was generated at or after the final-test start of 2026-06-09 15:12:25 UTC; the
latest evaluated timestamp was 2026-06-09 15:11:49 UTC.

The nonlinear result is promising evidence of structured deployment-time selection, but it is not
yet a profitable replica strategy. The creator-history family is retained, with the declining
late-fold delta recorded as a risk. The first unweighted logistic run is preserved in the
experiment ledger with an erratum rather than silently deleted.

## June outcome-source audit

The 2,840,322,917-byte June market-cap candle file is downloaded and SHA-256 verified as
`99c0546bd1fae9ae26dc505151200af5bfdef01f5a04af8c0251e33c77b59400`. Its 60,109,034
rows cover 764,451 tokens at one-second resolution. There are no nulls or duplicate
`(token_address, resolution, candle_time_ms)` keys.

The source is accepted only for outcome labels and coarse second-level sensitivity, with mandatory
filters for 36 pre-deployment rows across 35 tokens and one invalid OHLC row. It is forbidden as an
entry feature. Because it lacks block slot and transaction position, it cannot honestly provide
the required exact 0/1/2-slot execution prices by itself. No final-holdout result was evaluated.

## June exact-slot trade source audit

The 17,944,584,022-byte June trade file is downloaded and SHA-256 verified as
`ac10521276d8678fe7d2ce56649512c4dc274babdf0abd4f2d7df81a3b3cc03c`. The full audit
confirmed 133,978,933 rows, 767,195 tokens, zero duplicate event IDs, zero pre-deployment trades,
and consistent per-token deployment context.

The source contains block slot, transaction index, event index, deployment position, and USD/SOL
price fields, so exact 0/1/2-slot price construction is supported. Post-deployment same-slot
coverage is 891,400 trades across 250,787 tokens; slot+1 and slot+2 contain 382,126 and 382,781
trades respectively. The 734,865 events from the deployment transaction itself must be excluded
as non-replicable. Exactly 8,952 rows have null amount fields and require an explicit size/volume
filter, although their price fields remain present.

The next priority is a deterministic 0/1/2-slot entry-price table with explicit coverage and
missingness, followed by train/validation backtesting with fees and drawdown. The 429 GiB raw blocks
remain out of scope, and the final holdout remains sealed.

## Exact-slot entry-price proxy

A deterministic table now covers all 38,111 June classification tokens at requested delays 0, 1,
and 2, producing 114,333 unique token-delay rows. Classification and trade deployment context
matches exactly. Output SHA-256 is
`8971bc3c210d6d8a7b98b21fd6a0a66eefcc2aaf5b4e8b2fd27c6221d9cdad83`.

Coverage is 87.22%, 86.94%, and 86.34% for requested delays 0, 1, and 2. Only 13,836, 11,356,
and 12,100 rows respectively execute on the exact target slot; the remaining covered rows use the
first later observed trade. Median additional wait is one slot, but the p95 is 145, 154, and 176
slots. Consequently, the table is retained as an observed-trade execution proxy, not evidence that
every requested delay fills on time. Backtests must report actual delay and exact-target coverage.

## Six-second validation replica backtest

The training-only wallet history freezes a six-second median hold, USD 201.16 median first-buy
notional, 730.22 bps median round-trip fee, and 1,862.05 bps p90 fee. At the creator-history
validation operating point (PR-AUC 0.08934, precision 12.74%, recall 23.49%, F1 0.165), 387
sampled June validation deployments were selected. No prediction or outcome reached the final
holdout start; the latest backtest mark was 2026-06-09 14:40:28 UTC.

With the median fee, the all-observed 0-slot proxy has +33.08% population-weighted mean return,
+15.42% unweighted median return, 43.92% weighted hit rate, and 11.47% maximum drawdown. Exact
target-slot execution covers only 76.42% of the weighted attempts. Requested delay 1 falls to
-1.47% mean and -11.73% median after median fees. Requested delay 2 has a misleading +26.66%
mean but -9.74% median; rare winners dominate, and the tight capital model crosses zero for both
delayed strategies.

This supports the fixed-hold hypothesis on development data only for near-zero-slot execution.
The exit is still a last-trade mark rather than a demonstrated sell fill, the validation-selected
threshold is not independent, and the population-weighted sampled path is approximate. The final
holdout remains sealed. The next priority is to audit zero-slot transaction-order feasibility and
construct a sell-side executable exit proxy before freezing a candidate.

## Zero-slot position-lag falsification

The preceding first-trade zero-slot result is now classified as an optimistic price upper bound,
not an executable candidate. In 7,377 training-period same-slot target-wallet buys, the median
transaction-position gap after deployment is 112 (p10 25, p90 421). The prior proxy instead used
a median gap of one among its exact-same-slot validation fills; only 22.97% of those first trades
occurred at or after the frozen 112-position lag.

Using the first same-slot trade no earlier than deployment index plus 112 reduces population-
weighted fill coverage to 57.51%. With median training fees, weighted mean return falls from
+33.08% to +5.51%, while the unweighted median becomes -6.78%, hit rate is 38.18%, and maximum
drawdown is 111.97% with negative interim equity. At p90 fees, mean return is -5.81%, median is
-18.10%, and drawdown is 258.96%.

The position-lag hypothesis is rejected on validation and will not receive final-holdout access.
This negative result materially weakens a raw-return story but strengthens the submission's
honesty: profitable replication is dominated by execution timing, and same-slot feasibility must
be modeled rather than assumed. The next priority is a sell-side executable exit audit and a
head-to-head target-wallet comparison that does not reuse first-trade prices as attainable fills.
