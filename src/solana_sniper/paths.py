import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SOLANA_SNIPER_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "generated"
MANIFEST_PATH = PROJECT_ROOT / "experiments" / "manifest.jsonl"


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
