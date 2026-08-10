"""Check the minimum Graph-RMS runtime and report optional baseline support."""
from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED = ("numpy", "scipy", "sklearn", "torch", "pandas", "yaml", "psutil")
OPTIONAL = ("skimage", "skfuzzy", "igraph", "leidenalg", "matplotlib")


def version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "installed-version-unreported"))


def main() -> int:
    import torch

    required = {name: version(name) for name in REQUIRED}
    optional = {name: version(name) for name in OPTIONAL}
    try:
        import graphrms  # noqa: F401
        import graphrms.prototype  # noqa: F401
        core_import = True
    except Exception as exc:
        core_import = f"{type(exc).__name__}: {exc}"
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "required": required,
        "optional": optional,
        "graphrms_import": core_import,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(report, indent=2))
    missing = [name for name, value in required.items() if value is None]
    if missing or core_import is not True:
        print(f"Environment check failed; missing/broken: {missing or core_import}", file=sys.stderr)
        return 1
    print("Core Graph-RMS environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

