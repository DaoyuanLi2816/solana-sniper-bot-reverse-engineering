# Campaign status

Last verified: 2026-08-04 06:30 UTC

## Remote state

- Competition: `solana-sniper-bot-reverse-engineering`
- Entered and rules accepted: yes
- Final deadline: 2026-08-14 21:00 UTC
- Submission format: one Kaggle Writeup per team; it may be edited and re-saved before the
  deadline, but drafts and unsubmitted Writeups are not judged
- Authenticated API team count: 4
- Visible submitted projects: 4, all hidden until Hackathon close
- Our account: no Writeup created or submitted yet
- Discussion: two existing topics; no new technical or submission guidance

The published prize and team-limit text is inconsistent. The Overview advertises USD 1,000 total,
while the competition-specific Rules page still contains sponsor placeholders and lists a
USD 600/300/100 split. The API/UI permits ten team members while the Rules say five; the project
uses the stricter five-person limit. Competition data may not be publicly redistributed.

## Current corrected evidence

All current model and selector claims use strict-history dataset SHA-256
`57af874b0768eaf43c54f04d698cda9e3c3e1d9bf3e1a25c7c69910ecbe8817f`:

- 218,350 rows and 218,350 unique tokens
- 15,927 positive deployments and 202,423 systematically sampled negatives
- zero UTC-hour mismatches, weekday mismatches, and strict-history violations
- final holdout starts `2026-06-09T15:12:25Z` and remains sealed

The frozen raw signer-delta plus strict prior-signer-history model records standard-validation
PR-AUC 0.07028, precision 0.1203, recall 0.2011, and F1 0.1506 at threshold 0.937099. Adding the
four strict-history fields improves PR-AUC by +0.03280, +0.02366, and +0.01836 in the three
expanding development folds. Strict history is the top feature family in all three folds, narrowly
ahead of deployment-transaction structure.

Target-wallet behavior is descriptive, not a model feature. Across 15,921 matched closed tokens,
net PnL after recorded gas and DEX fees is USD 922,595, mean ROI +13.45%, median ROI +4.78%, and
hit rate 59.39%. In the strictly pre-holdout June window, 1,372 target buys have mean ROI +10.71%,
median +3.05%, hit rate 56.63%, and realized drawdown 6.89%.

## Corrected execution evidence

The training-only wallet behavior freezes a six-second hold, USD 201.16 notional, 730.22 bps
median round-trip fee, and 112-transaction same-slot position lag.

The optimistic first-observed-trade proxy reports, at median fees:

| requested delay | weighted mean | unweighted median | hit rate | max drawdown | solvent |
|---:|---:|---:|---:|---:|:---:|
| 0 | +74.86% | +16.08% | 49.43% | 3.20% | yes |
| 1 | -4.32% | -10.25% | 30.92% | 233.08% | no |
| 2 | +32.86% | -8.47% | 28.13% | 8.29% | yes |

Only requested delay zero passes the predeclared positive-mean, positive-median, solvent criterion.
After replacing its first-trade entry with the target's training-derived transaction position,
population-weighted coverage is 57.03%. Median-fee mean return is +8.11%, but median return is
-5.56%, hit rate 43.53%, drawdown 91.96%, and the capital path crosses zero. The executable
position-lag candidate is rejected; the raw zero-slot result is only an optimistic upper bound.

## Submission-first checklist

- [x] Corrected transaction-position evidence rerun twice and frozen
- [x] Corrected target-versus-replica comparison rerun twice and frozen
- [x] Writeup rewritten under 3,000 words with only corrected claims
- [x] Stale campaign status removed
- [x] Cover image generated and checked
- [ ] Public reproducibility notebook created and run end to end
- [ ] Public repository created with lock, commands, hashes, and negative results
- [ ] Kaggle Track, Media Gallery, Public Notebook, and Project Link attached
- [ ] Initial Writeup submitted and remote ID, status, and URL recorded

Current local commit containing the corrected execution evidence is `60bcae0`. The submission is
not yet valid until all unchecked items are complete and Kaggle shows `Submitted`.
