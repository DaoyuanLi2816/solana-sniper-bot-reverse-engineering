import pyarrow as pa
import pytest

from solana_sniper.audit_trades_metadata import (
    EXPECTED_BYTES,
    EXPECTED_ROWS,
    EXPECTED_SCHEMA,
    supports_exact_slot_delay,
    validate_metadata,
)


def expected_arrow_schema() -> pa.Schema:
    types = {
        "string": pa.string(),
        "int64": pa.int64(),
        "int32": pa.int32(),
        "double": pa.float64(),
    }
    return pa.schema([(name, types[kind]) for name, kind in EXPECTED_SCHEMA])


def test_trade_metadata_contract_supports_exact_slot_delay() -> None:
    schema = expected_arrow_schema()
    validate_metadata(schema, EXPECTED_ROWS, EXPECTED_BYTES)
    assert supports_exact_slot_delay(schema)


def test_trade_metadata_contract_rejects_missing_position() -> None:
    full_schema = expected_arrow_schema()
    schema = full_schema.remove(full_schema.names.index("tx_index"))
    assert not supports_exact_slot_delay(schema)
    with pytest.raises(ValueError, match="Unexpected trades schema"):
        validate_metadata(schema, EXPECTED_ROWS, EXPECTED_BYTES)


def test_trade_metadata_contract_rejects_remote_size_change() -> None:
    with pytest.raises(ValueError, match="Unexpected remote size"):
        validate_metadata(expected_arrow_schema(), EXPECTED_ROWS, EXPECTED_BYTES + 1)
