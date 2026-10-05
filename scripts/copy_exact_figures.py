#!/usr/bin/env python3
"""Copy the exact committed paper figures into outputs/."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIGURES = ROOT / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ["Framework.pdf", "top10.pdf", "USGS_analysis.pdf"]:
        src = FIGURES / name
        dst = OUT / name
        shutil.copy2(src, dst)
        print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
