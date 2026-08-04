# Reverse-Engineering a Zero-Block Solana Sniper: Selection Signal Is Not Execution Alpha

## Executive summary

The target wallet is unusually fast: 79.63% of its first buys occur in the deployment slot, its
median first entry is USD 184, and its median observed hold is six seconds. Yet the strongest lesson
from this project is not that a fast classifier automatically becomes a profitable bot. It is that
three questions must be separated:

1. What behavior does the target wallet exhibit?
2. Which fields available at token deployment explain its token selection?
3. Can a new trader obtain executable entries and exits after latency and fees?

We answer the first two positively. A leak-free, time-split model finds stable selection signal in
strictly prior deployment-signer history and the deployment transaction itself. Adding four strict
history features improves population-adjusted PR-AUC in all three expanding development folds:
+0.03280, +0.02366, and +0.01836. Standard validation PR-AUC is 0.07028 at an operating point of
12.03% precision, 20.11% recall, and 0.1506 F1.

The third answer is deliberately more cautious. An optimistic first-observed-trade backtest looks
excellent only at requested zero-slot latency. Once entry is moved to the target wallet's actual
training-derived transaction position, the typical trade loses money after fees and the tight
capital path becomes insolvent. We therefore present a selection model and an executable-feasibility
falsification, not a claim of production-ready profit.

## 1. Decision boundary and data integrity

We define `t_decision` as the token deployment transaction. Every classifier and entry-selection
feature must exist when that transaction is observed. Permitted inputs include transaction index,
fees, balances, instruction/account/log counts, metadata fields, UTC clock fields, and signer
deployment history strictly before the current block slot.

No post-deployment trade, candle, future price, migration, realized PnL, outcome label, or future
signer deployment enters the feature set. Post-deployment data is used only to define the target
label and to evaluate entry/exit marks. Same-slot and future signer history is excluded with a
strict smaller-slot window.

The corrected modeling table contains 218,350 unique tokens: 15,927 bought deployments and 202,423
systematically sampled non-buys. Every 25th negative deployment was sampled, and each sampled
negative receives weight 25 when computing population metrics. Accuracy and ROC-AUC are not used
as progress metrics under this imbalance; we report PR-AUC and threshold-specific precision,
recall, and F1.

The current strict-history table has SHA-256
`57af874b0768eaf43c54f04d698cda9e3c3e1d9bf3e1a25c7c69910ecbe8817f`, 218,350 unique
tokens, zero UTC-hour mismatches, zero weekday mismatches, and zero strict-history violations. An
earlier class-dependent clock-feature defect was detected, documented, repaired, and all claims
used here were rerun twice on the corrected table. The final chronological holdout begins at
`2026-06-09T15:12:25Z`; it remains sealed and no predictions were generated for it.

## 2. What the target wallet does

All 15,927 known bought deployments match to a first wallet buy. The first entry is in the
deployment slot for 79.63% of tokens. Although median slot and second latency are both zero, the
wallet is not buying at the first possible position: among same-slot buys, the full-period median
gap is 118 transaction positions after deployment.

Training-only behavior, cut off at `2026-05-18T16:44:18Z`, freezes the replica assumptions. The
median observed hold is six seconds, median first-buy notional is USD 201.16, median round-trip fee
is 730.22 bps, and p90 round-trip fee is 1,862.05 bps. Among 7,377 training-period same-slot buys,
the wallet's median transaction-position gap is 112, with p10 25 and p90 421. That large within-slot
gap is central to the execution audit below.

Recorded cash flows show that the target itself remains profitable after fees. Across 15,921
matched closed tokens, gross PnL is USD 1.468 million, recorded gas plus DEX fees are USD 545,896,
and net PnL is USD 922,595. Net mean ROI is +13.45%, median ROI +4.78%, and hit rate 59.39%.

For the strictly pre-holdout June comparison window, all 1,372 target outcomes close before the
boundary. Net mean ROI is +10.71%, median ROI +3.05%, hit rate 56.63%, and realized maximum
drawdown 6.89%. This curve books PnL at the final sell and does not measure intratrade mark-to-market
risk.

## 3. An interpretable deployment-time selector

The frozen model is histogram gradient boosting with class balancing. The core feature set combines
deployment-transaction structure with four strict prior-signer features:

- prior deployment count;
- prior active deployment-slot count;
- observed deployment age; and
- seconds since the previous deployment.

Against the same model without these four fields, standard validation PR-AUC rises from 0.05376 to
0.07028. Precision rises from 9.00% to 12.03% and F1 from 0.1261 to 0.1506, while recall changes
from 21.08% to 20.11%. The operating threshold is 0.937099. More importantly, history improves
PR-AUC in every predeclared expanding fold, including the latest fold, rather than succeeding only
on one convenient split.

Temporal permutation analysis reinforces the interpretation. Strict signer history is the
highest-ranked feature family in all three folds, with mean PR-AUC drop 0.05661, narrowly ahead of
transaction structure at 0.05262. Metadata and fee/compute groups are secondary; decision time has
mean drop -0.00029 and is not a useful story after UTC remediation.

The strongest individual signals are signer lamport delta, seconds since prior deployment,
observed signer deployment age, and prior deploy count. On standard validation, bought tokens have
median 8,249 seconds since the signer's previous deploy versus 149 for sampled non-buys, and median
observed signer age of 7.79 million seconds versus 1.98 million. The target appears to favor
deployment signers with a longer observed history and less immediate spam cadence, together with
a distinctive transaction-construction footprint.

This wording is intentionally precise: the observed `tx_signer` or fee payer is not proven to be
the token creator. Likewise, the fee-adjusted nonfee outflow
`max(-signer_lamport_delta - tx_fee_lamports, 0)` is a useful semantic proxy, but it can mix trade
principal, rent, funding, and protocol transfers. It stayed noninferior to raw signer delta but was
not a consistent PR-AUC improvement, so the raw field remains in the frozen selector.

## 4. Replica backtest: requested delay versus actual execution

The score threshold is selected on chronological validation. We then evaluate June development
deployments only, stopping strictly before the final holdout. Entry is the first observed trade at
or after deployment slot plus requested delay 0, 1, or 2, excluding the deployment transaction.
Exit is the last observed trade mark no later than entry plus six seconds. These are observed-price
proxies, not guaranteed size-aware fills.

Under the training median fee:

| requested delay | actual delay median / p90 | weighted mean | unweighted median | hit rate | max drawdown | solvent |
|---:|:---:|---:|---:|---:|---:|:---:|
| 0 | 0 / 1 | +74.86% | +16.08% | 49.43% | 3.20% | yes |
| 1 | 1 / 2 | -4.32% | -10.25% | 30.92% | 233.08% | no |
| 2 | 2 / 4 | +32.86% | -8.47% | 28.13% | 8.29% | yes |

The delayed results explain why mean return alone is unsafe. Delay two has a positive weighted mean
but a negative typical trade; a few right-tail winners dominate. Delay one is negative and
insolvent. Under the predeclared definition—positive weighted mean, positive unweighted median, and
solvency—only the optimistic requested-delay-zero proxy is viable.

But even that is not yet executable evidence. For the target wallet, the median same-slot entry is
112 transaction positions after deployment in the training period. We therefore replace the first
post-deployment trade with the first trade in the deployment slot at or after
`deploy_tx_index + 112`.

This position-aware policy attempts 335 selected candidates and fills 237, only 57.03% after
population weighting. At median fees, weighted mean return is +8.11%, but unweighted median is
-5.56%, hit rate 43.53%, maximum drawdown 91.96%, and the tight capital path briefly crosses zero.
At p90 fees, mean return falls to -3.21% and median to -16.88%.

The executable-position hypothesis is rejected. The first-trade zero-slot result is retained as an
optimistic price upper bound, not promoted as attainable alpha. The target-versus-replica comparison
makes the remaining gap visible: in the same pre-holdout window the target has +3.05% median ROI and
6.89% realized drawdown, while the position-lag replica has -5.56% median return and 91.96%
drawdown. Total dollar PnL is not comparable because target sizing is variable and replica sizing is
fixed and population weighted.

## 5. Negative results that changed the answer

We preserve rather than hide failed hypotheses:

- priority fee per consumed compute unit is rejected; it loses PR-AUC in two of three folds beyond
  the 0.002 noninferiority margin;
- fee-adjusted signer outflow is semantically interpretable but not a consistent performance gain;
- clock features lose their apparent importance after correcting the UTC construction defect;
- one- and two-slot delayed strategies fail the typical-trade viability test; and
- transaction-position lag falsifies the profitable zero-slot execution story.

These failures are useful reverse engineering. The classifier can identify launches that resemble
the wallet's choices, but the target's advantage also depends on within-slot execution, partial
exits, and infrastructure not recovered by deployment-time classification alone.

## 6. Reproducibility

The public notebook and repository reproduce the reported tables and figures without redistributing
competition data. Users attach the competition files under the documented paths, install the locked
`uv` environment, and run the audited commands. Each stage checks schema, row counts, unique keys,
time boundaries, and SHA-256 hashes. Every current experiment is appended to an immutable JSONL
ledger with parameters, source hashes, code parent, metrics, decision, and negative results.

The notebook marks `t_decision` explicitly before feature construction, demonstrates that
post-deployment tables are referenced only by label/backtest cells, and asserts that the maximum
evaluated development time precedes the sealed holdout. The final package includes all source code,
tests, environment lock, figures, and exact reproduction commands.

## Conclusion

The target wallet's token selection is partly recoverable from deployment-time evidence. The most
stable rule combines observed signer maturity and deployment spacing with signer balance commitment
and transaction structure. That signal is real under chronological, population-weighted validation.

The profitable strategy is not yet recovered. Honest execution modeling transforms a spectacular
first-trade backtest into a negative typical trade with severe drawdown. Our result is therefore a
leak-free behavioral and selection reverse engineering, plus a concrete demonstration that
zero-block Solana replication must solve transaction-position and executable-exit problems rather
than assume them away.
