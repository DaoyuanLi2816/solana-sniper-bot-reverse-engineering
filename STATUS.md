# Campaign status

Last verified: 2026-08-14 23:44 UTC

## Remote state

- Competition: `solana-sniper-bot-reverse-engineering`
- Entered and rules accepted: yes
- Final deadline: 2026-08-14 21:00 UTC
- Submission format: one Kaggle Writeup per team; it may be edited and re-saved before the
  deadline, but drafts and unsubmitted Writeups are not judged
- Authenticated API team count: 29
- Published Writeups after close: 29
- Our account: entered; authenticated API returns our Writeup as `ContentState.PUBLISHED`
- Discussion: two existing topics; no new technical or submission guidance

The structured Hackathon Track API resolves the prize as one first-place award of USD 1,000
(track `5836`, prize `9123`). The competition-specific Rules prose still contains an unfilled
sponsor placeholder and a stale USD 600/300/100 split. The API/UI permits ten team members while
the Rules say five; the project uses the stricter five-person limit. Competition data may not be
publicly redistributed.

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
- [x] Public reproducibility notebook created, run end to end, and audited for embedded data
- [x] Public repository created with lock, commands, hashes, and negative results
- [x] Kaggle public companion notebook version 1 completed and anonymously reachable
- [x] Kaggle Track attached (single track, auto-selected), repository and notebook links in the body
- [x] Writeup submitted 2026-08-05 and verified as `Submitted` on Kaggle

Corrected execution evidence is frozen in local commit `60bcae0`; the writeup and cover are frozen
in `2c2823f`; the executed notebook is frozen in `968b20b`. The public repository is
`https://github.com/DaoyuanLi2816/solana-sniper-bot-reverse-engineering`. The Kaggle companion is
`https://www.kaggle.com/code/distiller/solana-sniper-leakage-audited-reproduction` and version 1 is
`COMPLETE`.

## Submission

Submitted 2026-08-05 as
[Zero-Block Solana Sniper: Selection Signal Is Not Execution Alpha](https://www.kaggle.com/competitions/solana-sniper-bot-reverse-engineering/writeups/new-writeup-1785834990646).
Kaggle shows `Submitted`; the Writeups list shows the entry with its cover image under both
Your Work and the public All section. It remains editable until 2026-08-14 21:00 UTC, so any
correction found before then can still be published.

Live verification at 2026-08-11 02:09 UTC again showed `Submitted!`. The competition had 8 teams
and 7 submitted projects, with no new competition discussion requiring a writeup correction.

At 2026-08-11 11:13 UTC the writeup still showed `Submitted!`; the field had grown to 9 teams and
8 submitted projects. Discussion remained unchanged, so no evidence-backed writeup edit was made.

At 2026-08-11 20:25 UTC the writeup still showed `Submitted!`; the field had grown to 10 teams and
9 submitted projects. The rendered writeup retains both public links and contains no superseded
`5fc74...` hash. Discussion remained unchanged, so no evidence-backed writeup edit was made.

At 2026-08-12 23:30 UTC, after the in-app browser login expired, the authenticated Kaggle v2 API
independently confirmed that the account is entered and returned the expected submission as
`ContentState.PUBLISHED` (HackathonWriteUp `82840`, WriteUp `108478`, team `16635086`, track
`5836`, publish time `2026-08-05 06:30:53 UTC`). The field had grown to 11 teams and discussion
still contained only the two previously reviewed topics. No evidence-backed writeup edit was made.

At 2026-08-14 08:41 UTC, about 12 hours before the deadline, the authenticated API again returned
the same Writeup as `ContentState.PUBLISHED`. The field had grown to 18 teams. Competition-page
content hashes, data files, and discussion were unchanged; the public repository and notebook
returned HTTP 200 and the notebook kernel remained `COMPLETE`. The final holdout remains sealed,
and no evidence-backed writeup edit was made.

At 2026-08-14 17:44 UTC, about three hours before the deadline, the authenticated API again
returned the same Writeup as `ContentState.PUBLISHED`; the field had grown to 23 teams. The page,
data, discussion, public-artifact, notebook, test, format, and Git audits all passed unchanged.
This is the frozen final pre-deadline audit. No unreviewed model or Writeup changes are pending.

## Post-deadline audit

At 2026-08-14 23:44 UTC, the deadline had passed. The authenticated API returned all 29 team
Writeups as published, including ours with the expected IDs, title, URL, and track. No Writeup had
an awarded track-prize ID, so judging remains pending and no rank or prize claim is supported. The
public repository, notebook, and Writeup returned HTTP 200; the notebook kernel remained
`COMPLETE`. No experiment was started after the deadline and the final holdout remains sealed.

The strongest evidence, replica rejection, preserved negative results, and residual risks are
frozen in `reports/final_campaign_summary.md`.
