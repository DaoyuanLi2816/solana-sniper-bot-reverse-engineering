import pyarrow as pa
import pytest

from solana_sniper.audit_candles import (
    EXPECTED_SCHEMA,
    supports_exact_slot_delay,
    validate_source_metadata,
)


def expected_arrow_schema() -> pa.Schema:
    types = {"string": pa.string(), "int64": pa.int64(), "double": pa.float64()}
    return pa.schema([(name, types[kind]) for name, kind in EXPECTED_SCHEMA])


def test_validate_source_metadata() -> None:
    validate_source_metadata(expected_arrow_schema(), 60_109_034)


def test_validate_source_metadata_rejects_wrong_row_count() -> None:
    with pytest.raises(ValueError, match="Unexpected candle row count"):
        validate_source_metadata(expected_arrow_schema(), 1)


def test_exact_slot_delay_requires_slot_and_position() -> None:
    assert not supports_exact_slot_delay(expected_arrow_schema())
    assert supports_exact_slot_delay(
        expected_arrow_schema()
        .append(pa.field("block_slot", pa.int64()))
        .append(pa.field("transaction_index", pa.int64()))
    )
