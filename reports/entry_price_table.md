# Exact-slot entry-price proxy table

## Hypothesis and definition

After excluding every event in the deployment transaction, the first observed trade at or after a
target slot provides a deterministic, auditable entry-price proxy for requested delays of 0, 1,
and 2 slots.

For token deployment position `(deploy_slot, deploy_tx_index)` and requested delay `d`, the builder
sets `target_slot = deploy_slot + d`, filters to trades strictly after the deployment transaction,
and chooses the minimum fixed-width `slot_index_id` at or after the target. If the target slot has
no observed trade, the row records the later actual execution slot and the additional wait. If no
later trade exists, every execution field remains explicitly null.

This table is post-decision backtest data. It contains no class label, model prediction, or
selection feature and is forbidden from classifier inputs.

## Validated output

- Universe: 38,111 unique June classification tokens.
- Output: 114,333 unique `(token_address, requested_delay_slots)` rows.
- Classification/trade deployment time, slot, and transaction-index mismatches: zero.
- Output SHA-256: `8971bc3c210d6d8a7b98b21fd6a0a66eefcc2aaf5b4e8b2fd27c6221d9cdad83`.
- Classification input SHA-256:
  `5fc74ffb4d7ac5a0fd26ff5a4cb4326a89d227d273fd3c583ea88d827a018f12`.
- Trade input SHA-256: `ac10521276d8678fe7d2ce56649512c4dc274babdf0abd4f2d7df81a3b3cc03c`.

| Requested delay | Covered | Missing | Exact target slot | Later observed trade | Median extra wait | P95 extra wait |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 33,240 (87.22%) | 4,871 | 13,836 | 19,404 | 1 slot | 145 slots |
| 1 | 33,133 (86.94%) | 4,978 | 11,356 | 21,777 | 1 slot | 154 slots |
| 2 | 32,906 (86.34%) | 5,205 | 12,100 | 20,806 | 1 slot | 176 slots |

The source contains no trades for 3,429 universe tokens. Additional missing rows arise when a token
has trades but none at or after the requested target slot.

## Limitation retained for the backtest

The table proves deterministic position selection, not universal execution at the requested slot.
Most covered rows at each delay use a later observed trade, and the wait distribution has a long
tail. A backtest must therefore report both requested delay and actual delay; describing every row
as a 0/1/2-slot fill would be false. Results should separately show exact-target coverage and the
later-trade proxy, or restrict the primary scenario to exact-target observations and report the
resulting coverage selection risk.

The first focused SQL test failed because a derived entry-ID alias was joined with `USING` as if it
were a source column. The join was corrected to an explicit equality and all focused tests passed
before the full build. An earlier schema preflight also used `block_time` instead of the actual
classification column `blockTime`; it generated no artifact and was corrected before construction.

No prediction, return, PnL, hit rate, or final-holdout metric was generated in this step.
