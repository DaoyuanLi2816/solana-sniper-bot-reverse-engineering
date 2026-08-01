import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import yaml

from solana_sniper.paths import PROJECT_ROOT, RAW_DIR

CONFIG_PATH = PROJECT_ROOT / "config" / "competition.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "solana-sniper-campaign/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        expected_header = response.headers.get("Content-Length")
        expected = int(expected_header) if expected_header is not None else None
        remaining = expected
        while remaining is None or remaining > 0:
            requested = 8 * 1024 * 1024 if remaining is None else min(8 * 1024 * 1024, remaining)
            chunk = response.read(requested)
            if not chunk:
                break
            output.write(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    if expected is not None and temporary.stat().st_size != expected:
        raise OSError(
            f"Incomplete download for {destination.name}: {temporary.stat().st_size} != {expected}"
        )
    temporary.replace(destination)
    return {
        "file": destination.name,
        "url": url,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    sources = config["data_sources"]
    base_url = sources["wallet_base_url"].rstrip("/")
    records = []
    for filename in sources["wallet_files"]:
        destination = RAW_DIR / "wallet" / filename
        if destination.exists() and destination.stat().st_size > 0:
            records.append(
                {
                    "file": filename,
                    "url": f"{base_url}/{filename}",
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                    "reused": True,
                }
            )
            continue
        records.append(download_file(f"{base_url}/{filename}", destination))

    manifest = {
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "records": records,
    }
    manifest_path = RAW_DIR / "wallet" / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
