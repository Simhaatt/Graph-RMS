"""Create a clean ZIP and SHA-256 file from the current repository tree."""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".venv", "outputs"}
EXCLUDED_SUFFIXES = {".pyc", ".log", ".aux", ".out", ".zip"}


def include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return not any(part in EXCLUDED_PARTS for part in relative.parts) and path.suffix not in EXCLUDED_SUFFIXES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "Graph-RMS-v1.0.0.zip")
    args = parser.parse_args()
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and include(path):
                archive.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    checksum = sha256(args.output)
    checksum_path = args.output.with_suffix(args.output.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {args.output.name}\n", encoding="ascii")
    print(f"Created {args.output}\nSHA-256 {checksum}\nChecksum file {checksum_path}")


if __name__ == "__main__":
    main()

