# Target-wallet fee-adjusted PnL and replica comparison

## Hypothesis and cash-flow audit

This run tests one hypothesis: the target wallet's apparent profitability survives subtraction of
the recorded network and DEX fees. It does not change or tune the replica classifier or strategy.
Post-deployment cash flows are behavior and evaluation data only; they are not entry features.

The wallet source has 87,004 buy/sell rows and 87,004 unique transaction hashes, so no transaction
fee is duplicated across trade rows. Net PnL uses:

`sell receipts - buy costs - gas_usd - dex_usd`

Priority fee and tip are not added separately. They are components of `gas_native`: after
subtracting `priority_fee + tip_fee`, every row has a nonnegative residual (median 0.000005 SOL).
Adding them again would double-count the main buy-side execution cost.

A token is treated as closed only when it has a buy, a sell, and an observed close flag. The
median sold/bought token-amount ratio is 1.0. Of the 15,927 wallet tokens matched to known
deployments, 15,921 satisfy the closed-token rule. Another 240 wallet tokens cannot be placed in
the deployment-time split and are excluded from split comparisons.

## Full-period Part 1 description

The following is descriptive competitor analysis across all matched closed tokens, not model
selection and not replica final-holdout evaluation.

| metric | value |
|:---|---:|
| Closed tokens | 15,921 |
| Gross buy capital | $4,200,092 |
| Gross PnL | $1,468,491 |
| Recorded network fees | $435,403 |
| Recorded DEX fees | $110,492 |
| Net PnL | $922,595 |
| Net mean / median ROI | +13.45% / +4.78% |
| Net hit rate | 59.39% |
| Average win / loss | +$116.98 / -$28.37 |
| Median / p90 hold | 6 / 17 seconds |
| Median / p90 sell transactions | 4 / 8 |
| Realized max drawdown | 0.84% |

Recorded fees consume 37.17% of gross PnL, but the target remains profitable in aggregate, mean,
median, and hit rate. The small drawdown is a closed-token realized curve: PnL is booked at the
last sell, so it does not measure adverse intratrade marks.

## Pre-holdout development comparison

For a clean comparison, target-wallet deployments are restricted to `2026-06-01 00:00 UTC`
through the final-holdout start `2026-06-09 15:12:25 UTC`. All 1,372 target tokens in this window
close before the boundary; the latest outcome is `14:59:17 UTC`, leaving zero censored tokens.

| metric | target wallet, recorded fees | position-lag replica, median fees |
|:---|---:|---:|
| Entry rows | 1,372 actual buys | 275 sampled fills / weight 1,451 |
| Net mean ROI | +10.71% | +5.51% |
| Net median ROI | +3.05% | -6.78% |
| Hit rate | 56.63% | 38.18% |
| Max drawdown | 6.89% | 111.97% |
| Net PnL | $48,848 actual | $16,087 population-weighted proxy |

Target fees in this window total $41,866, reducing gross PnL from $90,714 to $48,848. The target's
average winning token earns $80.29; its average losing token loses $22.76. Median hold shortens to
four seconds, while p90 is 14 seconds.

The comparison also reports selection overlap at the frozen classifier operating point:
precision 12.74%, recall 23.49%, and F1 0.165. The replica does not yet match the competitor's
typical trade or risk path.

![Target versus replica development metrics](figures/competitor_fee_pnl.svg)

## Decision and limitations

The hypothesis is **supported**: recorded gas and DEX fees substantially reduce returns but do not
remove target-wallet profitability in either the full descriptive sample or the strictly
pre-holdout development window. This is evidence about the competitor, not evidence that the
replica can obtain the same fills.

Target PnL uses actual variable sizing and actual entries. Replica PnL uses a fixed notional,
sampled negatives with population weights, and a position-lag entry proxy; total dollars are
therefore not directly comparable. Cost fields are accepted as supplied without an independent
on-chain cash reconciliation. Realized drawdown omits intratrade mark-to-market losses. The
classifier and replica final chronological holdouts remain sealed.

The reproducible aggregate is `reports/generated/competitor_fee_pnl.json` (SHA-256
`d780e56c845dfe77e434ee43aeb03f7e622f4894b775a9f0ebe68417294ea3e9`). The figure SHA-256 is
`b2a86e1f2312e3bd4d65d220c05ff4dc220323561983845225270e1fcdbd0dcd`.
