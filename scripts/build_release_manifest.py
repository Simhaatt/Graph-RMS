"""Write SHA-256 and size for every curated release file."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/provenance/release_file_manifest.csv"
EXCLUDE = {OUTPUT, ROOT / "Graph-RMS-v1.0.0.zip"}
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "outputs"}
EXCLUDED_SUFFIXES = {".pyc", ".log", ".aux", ".out", ".zip"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    records = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if (not path.is_file() or path in EXCLUDE
                or any(part in EXCLUDED_PARTS for part in relative.parts)
                or path.suffix in EXCLUDED_SUFFIXES):
            continue
        records.append({
            "relative_path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} release checksums to {OUTPUT}")


if __name__ == "__main__":
    main()
