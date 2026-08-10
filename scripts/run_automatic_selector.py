"""Run automatic-v2 selection while preserving selection/evaluation separation."""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "scripts/_run_automatic_selection.py"), run_name="__main__")

