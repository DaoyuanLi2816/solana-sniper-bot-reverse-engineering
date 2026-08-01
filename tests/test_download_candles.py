import pytest

from solana_sniper.download_candles import parse_content_range, validate_download_budget


def test_parse_content_range() -> None:
    assert parse_content_range("bytes 40-99/100") == (40, 99, 100)
    assert parse_content_range("bytes 99-40/100") is None
    assert parse_content_range("items 40-99/100") is None
    assert parse_content_range(None) is None


def test_download_budget_accounts_for_partial_file() -> None:
    assert (
        validate_download_budget(
            expected_bytes=100,
            partial_bytes=40,
            free_bytes=80,
            reserve_bytes=20,
        )
        == 60
    )


def test_download_budget_rejects_insufficient_space() -> None:
    with pytest.raises(OSError, match="Insufficient disk budget"):
        validate_download_budget(
            expected_bytes=100,
            partial_bytes=40,
            free_bytes=79,
            reserve_bytes=20,
        )


def test_download_budget_rejects_oversized_partial() -> None:
    with pytest.raises(ValueError, match="outside the expected file bounds"):
        validate_download_budget(
            expected_bytes=100,
            partial_bytes=101,
            free_bytes=1_000,
            reserve_bytes=20,
        )
