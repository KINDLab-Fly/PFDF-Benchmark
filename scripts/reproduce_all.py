#!/usr/bin/env python3
"""Recreate tables from saved results and copy exact archived figures into outputs/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script: str) -> None:
    subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=ROOT, check=True)


def main() -> None:
    run("make_table1_weather_prompt.py")
    run("make_table2_baseline.py")
    run("make_table3_region_improvement.py")
    run("copy_exact_figures.py")
    run("verify_artifacts.py")
    run("verify_published_predictions.py")
    print("\nDone. Reproduced artifacts are in outputs/.")


if __name__ == "__main__":
    main()
