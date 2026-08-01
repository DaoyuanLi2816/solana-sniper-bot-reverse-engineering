"""Stream selected members from the uncompressed 41.5 GB core tar archive.

The source server does not support HTTP range requests. Streaming mode avoids
storing the full archive and can stop as soon as all requested members have
been extracted. Never request the 429 GB optional raw-block dataset here.
"""

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import yaml

from solana_sniper.paths import PROJECT_ROOT, RAW_DIR

CONFIG_PATH = PROJECT_ROOT / "config" / "competition.yaml"
SAFE_INITIAL_MEMBERS = {
    "bought_deploy_txs.jsonl.gz",
    "bought_deploy_txs_index.parquet",
    "bought_deployers_activity.parquet",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_selected(url: str, members: set[str], destination: Path) -> list[dict[str, object]]:
    destination.mkdir(parents=True, exist_ok=True)
    missing = {name for name in members if not (destination / name).exists()}
    if not missing:
        return [
            {
                "file": name,
                "bytes": (destination / name).stat().st_size,
                "sha256": sha256_file(destination / name),
                "reused": True,
            }
            for name in sorted(members)
        ]

    request = urllib.request.Request(url, headers={"User-Agent": "solana-sniper-campaign/0.1"})
    extracted: list[dict[str, object]] = []
    with (
        urllib.request.urlopen(request, timeout=300) as response,
        tarfile.open(fileobj=response, mode="r|") as archive,
    ):
        for member in archive:
            name = Path(member.name).name
            if name not in missing or not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Could not read archive member {member.name}")
            target = destination / name
            temporary = target.with_suffix(target.suffix + ".part")
            with temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            if temporary.stat().st_size != member.size:
                raise OSError(f"Incomplete archive member {name}")
            temporary.replace(target)
            extracted.append(
                {
                    "file": name,
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )
            missing.remove(name)
            if not missing:
                break
    if missing:
        raise FileNotFoundError(f"Archive members not found: {sorted(missing)}")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--member",
        action="append",
        dest="members",
        choices=sorted(SAFE_INITIAL_MEMBERS),
        help="Safe initial member to extract; may be repeated.",
    )
    args = parser.parse_args()
    selected = set(args.members or SAFE_INITIAL_MEMBERS)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    url = config["data_sources"]["core_archive_url"]
    records = extract_selected(url, selected, RAW_DIR / "core")
    manifest = {
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "source": url,
        "records": records,
    }
    path = RAW_DIR / "core" / "download_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
