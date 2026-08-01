# June trade source audit

## Hypothesis

The bounded June `pumpfun_trades.parquet` artifact contains enough transaction-position data to
construct honest 0/1/2-slot replica entry prices without downloading the approximately 429 GiB
raw-block archive. It remains strictly post-decision data and is forbidden from the selection
model's features.

## Result

The hypothesis is accepted. A metadata-only Range audit first confirmed the required fields before
the 17,944,584,022-byte body was downloaded. The final file has SHA-256
`ac10521276d8678fe7d2ce56649512c4dc274babdf0abd4f2d7df81a3b3cc03c`.

- Full scan: 133,978,933 rows, 552 row groups, and 767,195 unique tokens.
- Exact position fields: `block_slot`, `tx_index`, `event_index`, `deploy_block_slot`, and
  `deploy_tx_index`, with both USD and SOL prices.
- Zero duplicate `slot_index_id` values and zero inconsistent per-token deployment contexts.
- Zero rows before their deployment slot/transaction position and zero rows before deployment
  time.
- Zero timestamp, slot-index encoding, side, program, price, decimal, or quote-mint violations.
- The metadata-tail SHA-256 is
  `718d442b295d7c000bc79a7eba9a56efea749299d0bb7295a9dafedac82cb6e6`.

The source has substantial exact-delay coverage:

- 891,400 post-deployment same-slot trades across 250,787 tokens.
- 382,126 slot+1 trades across 188,000 tokens.
- 382,781 slot+2 trades across 237,728 tokens.

There are 734,865 events in the deployment transaction itself. A replica cannot react to a token
deployment and then enter earlier in that same transaction, so every entry-price builder must use
`block_slot > deploy_block_slot` or, within the deployment slot, `tx_index > deploy_tx_index`.

## Required filters and recorded caveats

Exactly 8,952 rows have null `amount_usd` and `amount_sol`; all are `pump_amm` sells with zero
`quote_amount`. Their price fields are present, so they may remain in price-only sequencing, but
must be excluded from trade-size or volume calculations. Prices and base amounts are otherwise
strictly positive and no amount is negative.

`creator_address` differs from `deploy_tx_signer` on 681,040 rows. This is a legitimate semantic
difference, not a data-quality error, and the fields must not be treated as interchangeable.

The initial audit used the wrong `slot_index_id` field widths and formatted timestamps in the
machine's local timezone. Both guards were corrected to the observed 12/6/4 encoding and an
epoch-based UTC comparison, then the complete audit was rerun. These failed guard assumptions are
retained in the experiment ledger.

No candidate predictions, returns, PnL, or final-holdout metrics were generated. The next step is a
deterministic entry-price table for 0/1/2-slot delays, with same-transaction exclusion and explicit
coverage/missingness, before any strategy backtest is allowed.
