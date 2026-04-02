from __future__ import annotations

import sys
from pathlib import Path


def configure_local_imports() -> None:
    script_dir = Path(__file__).resolve().parent
    chunky_root = script_dir.parent
    workspace_root = chunky_root.parent

    sys.path.insert(0, str(chunky_root / "src"))
    sys.path.insert(0, str(workspace_root / "hmmx" / "src"))


configure_local_imports()

from chunky.cli import main


if __name__ == "__main__":
    raise SystemExit(main())