#!/usr/bin/env python3
"""Recreate numerical plots; preserve the exact paper PDFs in figures/."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / "src"), MPLBACKEND="Agg")
    for module in ["experiments.plot_rlfs_selected_groups_top10_categories",
                   "experiments.plot_three_region_top_features_usgs_grid"]:
        subprocess.run([sys.executable, "-m", module], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
