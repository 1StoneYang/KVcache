#!/usr/bin/env python3
"""Run ReKV through the shared VLessHallu AMBER/CHAIR entry point."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VLESS = Path("/root/autodl-tmp/Method/VLessHallu/VLessHallu")
for path in (str(ROOT), str(VLESS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from vlesshallu.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
