# Post-deadline competitor audit

Read-only public audit refreshed at `2026-08-21T07:03:35Z`. No submitted artifact, model,
threshold, experiment, or sealed holdout was changed.

## Live result state

- Kaggle still presents our entry as `Submitted`; no winner or prize badge is visible.
- The public Writeups pages still contain only submitted entries, not an announced ranking.
- No Kaggle result, prize, compliance, or additional-material request was found in the connected
  mailbox after `2026-08-20`.
- Therefore judging remains pending. `userRank: 0`, when returned by Kaggle before ranking, is a
  placeholder and is not evidence of first place.

## Comparison scope and caveats

This is a qualitative audit of the strongest-looking public Writeups, not a reconstructed judge
score. Metrics below are authors' public claims and are not independently rerun here. They use
different universes, folds, operating points, fill models, and fee assumptions, so raw values are
not perfectly comparable. The official rubric also rewards behavioral insight, interpretability,
execution realism, and reproducibility rather than classification AP alone.

## Strong public competitors

| Entry | Public evidence | Assessment |
|---|---|---|
| **Six Seconds to Decide** | Full 5,076,421-candidate universe; corrected June PR-AUC `0.20471`, precision `29.32%`, recall `42.60%`, F1 `0.34736`; order-invariant same-timestamp history audit; source-built execution plus a secondary curve replay; median modeled execution remains negative. | The most complete visible entry and the clearest favorite: strong classification, explicit leakage corrections, full-universe coverage, and unusually careful execution caveats. |
| **The Sniper Remembers** | June PR-AUC `0.2388` on 852,083 launches; precision `34.03%`, recall `36.76%`, F1 about `0.353`; strict history and delay analysis; typical trade becomes negative despite positive tail means. | Probably a top contender on classification and interpretability. |
| **Reconstructing a Solana sniper from sealed wallet evidence** | One-time June average precision `0.222669`; balanced precision `29.42%`, recall `44.48%`, F1 `0.35415`; conservative precision `34.76%`; primary one-slot execution loses `5.2054 SOL`. | Strong sealed design and operating-point reporting. Reproducibility is weaker because strangers can reprint tables but cannot recompute June AP from the private prediction bundle. |
| **Deployer Memory** | June PR-AUC `0.2009`, precision `26.25%`, recall `45.86%`, F1 `0.3339`; same-slot ROI positive but one-slot ROI `-23.06%`; AMM-aware replay. | Strong classification and execution narrative; likely a serious podium candidate. |
| **118 Transactions Later** | Retrospective PR-AUC `0.1341` with an uncertainty interval; design-weighted precision `19.67%`, recall `27.41%`, F1 `22.90%`; 140 control clusters and explicit abstention from a profit claim. | Methodologically careful and highly credible, although it is retrospective rather than a pristine one-shot holdout. |
| **The Bot Has a Memory** | May PR-AUC `0.27616`; one-time June PR-AUC `0.10341`, precision `18.47%`, recall `26.79%`, F1 `0.21868`; controlled ablations and broad latency/fee stress tests. | Strong experimental discipline and presentation; lower June score than the leaders but still competitive. |
| **The edge is 400 milliseconds wide** | PR-AUC around `15.5x` prevalence; max-F1 precision `12.2%`, recall `23.7%`, F1 `0.161`; detailed fee ledger, slot-delay table, and target/replica comparison. | Similar honest conclusion to ours, but with a more comprehensive full-population and execution presentation. |
| **A sniper that bets in notches** | June PR-AUC `0.1017` after a large validation-to-test decline; best reported F1 `0.3294`; documents that market-cap ROI can overstate returns by `7.2x` to `16.6x`. | Strong economic critique and transparent correction history; prior-target-buy features could receive scrutiny as a circular behavior-cloning signal even if they are known before `t_decision`. |
| **Reverse-engineering the signal, then stress-testing the trade** | Full 5,076,421-launch universe; June AP `0.073965`; top-25/day precision `22.67%`; realistic one- and two-slot replays lose money; independent bit-for-bit rebuild. | Very strong reproducibility and honesty, with weaker overall classification recall/AP. |
| **Priced Into the Zero Block** | June PR-AUC about `0.0678`, precision `13.2%`, recall `11.4%`, F1 `0.123`; detailed 852,083-row June holdout and curve-level replay; replica loses after one slot. | Excellent execution analysis but classification quality is near ours and below the strongest entries. |

One flashy entry, **The Profitable Sniper**, reports PR-AUC `0.8561` only after restricting the
problem to tokens the target already bought and predicting their profitability. That is not the
required selection problem over all deployments and is not treated as a compliant classification
leader here.

## Our relative position

Our entry remains competitive on the parts judges may reward for scientific restraint:

- strict `t_decision` feature boundaries and an explicit leakage remediation trail;
- preserved negative results rather than post-hoc winner selection;
- exact reporting of requested delay, observed fill proxy, 112-transaction position lag, fees,
  insolvency, and drawdown;
- a public notebook and repository that reproduce the published artifacts;
- the defensible conclusion that selection signal is not executable alpha.

The major weaknesses are material:

- the classification table uses 202,423 systematically sampled negatives rather than the full
  5,076,421-candidate universe used by several rivals;
- the final chronological holdout stayed sealed, so PR-AUC `0.07028` and F1 `0.1506` are development
  validation estimates rather than an independent final score;
- several visible rivals report June PR-AUC around `0.20` to `0.24` with F1 around `0.33` to `0.35`;
- our position-aware replica is insolvent and has `-5.56%` median return at median fees, which is an
  honest result but not a positive strategy outcome.

## Calibrated conclusion

On public evidence, our submission should not be described as the favorite. Its rigor may keep it
competitive in a judged rubric, but the most plausible outcome is a respectable non-winning finish
or a place somewhere in the broader top group. A subjective pre-result range of roughly **top 5 to
top 10** is more defensible than a first-place expectation; this is not an observed rank.

The single advertised monetary prize means only first place pays. Until Kaggle assigns an award,
the correct campaign status is **valid published submission, judging pending, no prize claim**.
