# Final campaign summary

Verified after the deadline at `2026-08-14T23:44:45Z`; result state refreshed at
`2026-08-21T07:03:35Z`.

## Submission outcome

- Competition deadline: `2026-08-14T21:00:00Z`.
- Final field: 29 teams and 29 published Writeups.
- Our HackathonWriteUp: `82840`; WriteUp: `108478`; team: `16635086`; track: `5836`.
- Final submission state: `ContentState.PUBLISHED`.
- Public notebook: `distiller/solana-sniper-leakage-audited-reproduction`, kernel status
  `COMPLETE`.
- Public repository and notebook both returned HTTP 200 after the deadline.
- Awarded track-prize IDs were empty for every Writeup at this audit. Judging is pending; no rank,
  prize, or winning claim is supported yet.
- No experiment was started after the deadline, and the final chronological holdout remains sealed.
- The `2026-08-21` public-result and mailbox refresh still found no winner, prize, compliance, or
  additional-material notice. A public competitor audit is preserved in
  `reports/postdeadline_competitor_audit.md`.

## Strongest supported result

The strongest result is a leak-free deployment-time selection model, not a profitable executable
replica. The corrected table contains 218,350 unique deployments: 15,927 target buys and 202,423
systematically sampled negatives. All classifier features are available at `t_decision`, the token
deployment transaction. Strict prior-signer history is computed using smaller slots only.

Adding four strict-history fields to deployment-transaction features improves population-adjusted
PR-AUC in all three expanding development folds by `+0.03280`, `+0.02366`, and `+0.01836`. On the
standard chronological validation split, the frozen model has PR-AUC `0.07028`, precision
`0.1203`, recall `0.2011`, and F1 `0.1506` at threshold `0.937099`. Strict history is the leading
feature family in every fold, narrowly ahead of deployment-transaction structure.

The behavioral analysis is also strongly supported. Across 15,921 matched closed target-wallet
tokens, recorded-fee net PnL is USD 922,595, net mean ROI is `+13.45%`, median ROI is `+4.78%`, and
hit rate is `59.39%`. In the strictly pre-holdout June comparison window, 1,372 target buys have
mean ROI `+10.71%`, median ROI `+3.05%`, hit rate `56.63%`, and realized drawdown `6.89%`.

## Replica conclusion

The optimistic first-observed-trade proxy passes only at requested zero-slot delay. With training
median fees it reports weighted mean return `+74.86%`, unweighted median `+16.08%`, hit rate
`49.43%`, and drawdown `3.20%`. Requested one-slot delay is negative and insolvent; requested
two-slot delay has a positive weighted mean but a negative median.

Replacing the optimistic entry with the target's frozen training-period median position lag of 112
transactions rejects the executable hypothesis. Population-weighted fill coverage is `57.03%`.
At median fees, weighted mean return is `+8.11%`, but the unweighted median is `-5.56%`, hit rate is
`43.53%`, maximum drawdown is `91.96%`, and the tight capital path crosses zero. The profitable
zero-slot result is therefore retained only as an upper bound.

## Preserved negative results

- Priority fee per consumed compute unit loses PR-AUC beyond the noninferiority margin in two of
  three folds.
- Fee-adjusted signer outflow is more interpretable but is not a consistent PR-AUC improvement.
- Clock-feature importance disappears after the UTC construction defect is repaired.
- Requested one- and two-slot strategies fail the typical-trade viability criterion.
- Transaction-position lag falsifies the profitable executable-replica story.
- The earlier class-dependent clock defect and superseded dataset hash are documented and excluded
  from every submitted metric.

## Residual risks

- The final chronological holdout was deliberately never opened, so the reported model estimate is
  development validation rather than a final independent score.
- Transaction position is an execution proxy, not proof that a new system can observe and react in
  the same slot.
- Entry and six-second exit values are observed trade marks, not size-aware guaranteed fills.
- The observed signer or fee payer is not proven to be the token creator.
- Target-wallet fees are accepted from the provided fields without independent on-chain cash
  reconciliation; realized drawdown omits intratrade mark-to-market risk.
- Replica fixed sizing and population weighting are not directly comparable with the target's
  variable sizing and exact portfolio path.
- This is a judged hackathon with 29 published entries. Submission validity does not imply a prize.

## Frozen artifacts

- Corrected classification SHA-256:
  `57af874b0768eaf43c54f04d698cda9e3c3e1d9bf3e1a25c7c69910ecbe8817f`.
- Experiment ledger: 42 immutable JSONL entries; SHA-256
  `3e89744eacc34f208b832b341d3bdd000fee76fea87967938abecb8239ee214f`.
- Final pre-deadline repository commit: `afc5275fcfbd1a2998788ed85d1d84050b6bffe5`.
- Kaggle Writeup:
  `https://www.kaggle.com/competitions/solana-sniper-bot-reverse-engineering/writeups/new-writeup-1785834990646`.
- Public notebook:
  `https://www.kaggle.com/code/distiller/solana-sniper-leakage-audited-reproduction`.
- Public repository:
  `https://github.com/DaoyuanLi2816/solana-sniper-bot-reverse-engineering`.
