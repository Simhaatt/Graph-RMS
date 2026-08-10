"""Inventory the supplied parent workspace using relative, publication-safe paths."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = RELEASE_ROOT.parent


def classify(path: Path, workspace: Path) -> tuple[str, str]:
    relative = path.relative_to(workspace)
    parts = {part.lower() for part in relative.parts}
    suffix = path.suffix.lower()
    size = path.stat().st_size
    if relative.parts and relative.parts[0] == RELEASE_ROOT.name:
        return "included_release", "curated release artifact"
    if size >= 25 * 1024 * 1024:
        return "large_excluded", "at least 25 MiB; inspect manually before any archive"
    if suffix in {".mat", ".bsq", ".hdr", ".tif", ".tiff"} or "data" in parts:
        return "third_party_or_raw_data", "not redistributed; provider terms apply"
    if parts & {"__pycache__", "node_modules", ".cache", "tmp"} or suffix in {".pyc", ".log", ".aux", ".out"}:
        return "temporary_or_build", "cache, log, dependency, or build product"
    if suffix == ".zip":
        return "historical_archive", "transport/archive artifact; selected evidence extracted separately"
    if suffix == ".ipynb":
        return "historical_notebook", "development/compute notebook; not canonical release entry point"
    if suffix in {".py", ".json", ".yaml", ".yml", ".csv", ".md", ".txt", ".npy", ".png"}:
        return "supplied_code_or_output", "reviewed through frozen registry/evidence hierarchy"
    return "other_excluded", "not required by the curated reproducibility release"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=RELEASE_ROOT / "docs/WORKSPACE_INVENTORY.csv")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    records = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or path.resolve() == output:
            continue
        category, action = classify(path, workspace)
        records.append({
            "relative_path": path.relative_to(workspace).as_posix(),
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
            "category": category,
            "release_action": action,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Inventoried {len(records)} files to {output}")


if __name__ == "__main__":
    main()

