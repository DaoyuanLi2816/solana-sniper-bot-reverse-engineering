import json
import re
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import yaml

from solana_sniper.manifest import sha256_file
from solana_sniper.paths import PROJECT_ROOT, RAW_DIR

CONFIG_PATH = PROJECT_ROOT / "config" / "competition.yaml"
DESTINATION = RAW_DIR / "june" / "mcap_candles.parquet"
MANIFEST_PATH = RAW_DIR / "june" / "candles_download_manifest.json"
CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_BYTES = 256 * 1024 * 1024
RESERVED_FREE_BYTES = 10 * 1024 * 1024 * 1024
USER_AGENT = "solana-sniper-campaign/0.1"


def parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    if match is None:
        return None
    start, end, total = (int(part) for part in match.groups())
    if start > end or end >= total:
        return None
    return start, end, total


def validate_download_budget(
    *, expected_bytes: int, partial_bytes: int, free_bytes: int, reserve_bytes: int
) -> int:
    if expected_bytes <= 0:
        raise ValueError("Expected download size must be positive")
    if partial_bytes < 0 or partial_bytes > expected_bytes:
        raise ValueError("Partial download size is outside the expected file bounds")
    remaining_bytes = expected_bytes - partial_bytes
    if free_bytes - remaining_bytes < reserve_bytes:
        raise OSError(
            "Insufficient disk budget: "
            f"free={free_bytes}, remaining={remaining_bytes}, reserve={reserve_bytes}"
        )
    return remaining_bytes


def probe_remote(url: str) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
        method="HEAD",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        size_header = response.headers.get("Content-Length")
        if size_header is None:
            raise OSError("Remote source did not provide Content-Length")
        return {
            "url": url,
            "bytes": int(size_header),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "accept_ranges": response.headers.get("Accept-Ranges"),
        }


def download_with_resume(url: str, destination: Path) -> dict[str, object]:
    remote = probe_remote(url)
    expected_bytes = int(remote["bytes"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    if destination.exists():
        if destination.stat().st_size != expected_bytes:
            raise OSError(
                f"Existing final file has unexpected size: {destination.stat().st_size} "
                f"!= {expected_bytes}"
            )
        return {
            **remote,
            "file": destination.name,
            "sha256": sha256_file(destination),
            "reused": True,
            "source_role": "outcome_labels_and_backtest_only",
            "entry_feature_use_forbidden": True,
        }

    partial_bytes = temporary.stat().st_size if temporary.exists() else 0
    free_bytes = shutil.disk_usage(destination.parent).free
    validate_download_budget(
        expected_bytes=expected_bytes,
        partial_bytes=partial_bytes,
        free_bytes=free_bytes,
        reserve_bytes=RESERVED_FREE_BYTES,
    )

    headers = {"User-Agent": USER_AGENT}
    if partial_bytes:
        headers["Range"] = f"bytes={partial_bytes}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = response.status
        if partial_bytes:
            content_range = parse_content_range(response.headers.get("Content-Range"))
            if status != 206 or content_range is None:
                raise OSError("Remote source did not honor the resume Range request")
            start, _, total = content_range
            if start != partial_bytes or total != expected_bytes:
                raise OSError(
                    "Resume response does not match the local partial file: "
                    f"start={start}, total={total}, partial={partial_bytes}, "
                    f"expected={expected_bytes}"
                )
        elif status != 200:
            raise OSError(f"Unexpected initial download response status: {status}")

        mode = "ab" if partial_bytes else "wb"
        downloaded = partial_bytes
        next_progress = ((downloaded // PROGRESS_BYTES) + 1) * PROGRESS_BYTES
        with temporary.open(mode) as output:
            while chunk := response.read(CHUNK_BYTES):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_progress:
                    percent = 100.0 * downloaded / expected_bytes
                    print(
                        f"downloaded={downloaded}/{expected_bytes} ({percent:.1f}%)",
                        flush=True,
                    )
                    next_progress += PROGRESS_BYTES

    actual_bytes = temporary.stat().st_size
    if actual_bytes != expected_bytes:
        raise OSError(f"Incomplete download: {actual_bytes} != {expected_bytes}")
    temporary.replace(destination)
    return {
        **remote,
        "file": destination.name,
        "sha256": sha256_file(destination),
        "reused": False,
        "source_role": "outcome_labels_and_backtest_only",
        "entry_feature_use_forbidden": True,
    }


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    urls = config["data_sources"]["optional_post_decision_sources"]
    matches = [url for url in urls if url.endswith("/mcap_candles.parquet")]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one candle source, found {len(matches)}")

    record = download_with_resume(matches[0], DESTINATION)
    manifest = {
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "records": [record],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
