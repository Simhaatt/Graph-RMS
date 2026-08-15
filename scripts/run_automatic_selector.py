"""Run the recovered automatic-v2 selector with selection/evaluation separation."""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "scripts/_run_automatic_v2.py"), run_name="__main__")
