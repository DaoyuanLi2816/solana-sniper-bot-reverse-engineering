import json
from datetime import UTC, datetime

import yaml

from solana_sniper.download_candles import download_with_resume
from solana_sniper.paths import PROJECT_ROOT, RAW_DIR

CONFIG_PATH = PROJECT_ROOT / "config" / "competition.yaml"
DESTINATION = RAW_DIR / "june" / "pumpfun_trades.parquet"
MANIFEST_PATH = RAW_DIR / "june" / "trades_download_manifest.json"


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    urls = config["data_sources"]["optional_post_decision_sources"]
    matches = [url for url in urls if url.endswith("/pumpfun_trades.parquet")]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one trades source, found {len(matches)}")

    record = download_with_resume(matches[0], DESTINATION)
    manifest = {
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "records": [record],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
