# June market-cap candle source audit

## Hypothesis

The bounded June `mcap_candles.parquet` artifact can supply reproducible outcome labels and
coarse delay/return evidence without downloading the approximately 429 GiB raw-block archive.
It must remain post-decision data and must never enter the token-selection feature matrix.

## Result

The hypothesis is accepted with mandatory row filters for outcome labeling, but rejected for
exact 0/1/2-slot execution modeling.

- The remote server advertised and delivered exactly 2,840,322,917 bytes with Range support.
- Parquet metadata and the full scan both report 60,109,034 rows in 255 row groups.
- The file contains 764,451 unique token addresses and only the `1s` resolution.
- The time span is exactly 2026-06-01 00:00:00 through 2026-06-30 23:59:59 UTC.
- All twelve columns have zero nulls; `(token_address, resolution, candle_time_ms)` has no
  duplicate keys; all millisecond and second timestamp pairs agree.
- SHA-256 is `99c0546bd1fae9ae26dc505151200af5bfdef01f5a04af8c0251e33c77b59400`.

The full scan found two upstream quality exceptions that must not be silently retained:

- 36 rows across 35 tokens occur 1--10 seconds before their declared deployment time.
- One row has `open_mcap < low_mcap`; the maximum violation is USD 5.2794 of market cap.

Every outcome builder must therefore require `candle_time_ms >= deploy_time_ms` and valid OHLC
bounds. These are source-cleaning filters, not model features. The raw file remains immutable.

## Slot-delay limitation

The schema has neither `block_slot` nor transaction position. One-second candles cannot identify
the executable price at an exact 0-, 1-, or 2-slot delay, especially when multiple Solana slots
fall within one second. The source is suitable for labels and coarse second-level sensitivity,
but it is not accepted as the sole replica-backtest input. The next bounded step is a metadata-only
audit of the June trade artifact for slot and transaction-position fields before deciding whether
its approximately 16.71 GiB body is justified.

No candidate predictions, returns, PnL, or final-holdout metrics were generated in this audit.
