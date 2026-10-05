#!/usr/bin/env python3
"""Check immutable inputs, generated tables, and exact archived figures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "release-manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    table = pd.read_csv(ROOT / "outputs/table2_baseline_benchmark.csv")
    paper = (ROOT / "paper/physics_model.tex").read_text()
    paper_table = re.search(r"\\begin\{tabularx?\}.*?\\end\{tabularx?\}", paper, re.S)[0]
    cells = re.findall(r"(\d\.\d{3})\s*\\pm\s*(\d\.\d{3})", paper_table)
    assert len(cells) == 52
    expected = [tuple(map(float, c)) for c in cells]
    got = [tuple(map(float, value.split(" +/- "))) for value in table.iloc[:, 1:].to_numpy().flat]
    assert len(got) == len(expected)
    assert np.allclose(got, expected, rtol=0, atol=0.00050001)
    for name in ["Framework.pdf", "top10.pdf", "USGS_analysis.pdf"]:
        assert (ROOT / "figures" / name).read_bytes() == (ROOT / "outputs" / name).read_bytes()
    regional = pd.read_csv(ROOT / "outputs/table3_region_improvement.csv", keep_default_na=False, dtype=str)
    assert len(regional) == 10
    # The original paper prints +0.12 here, but its underlying CSV is 0.1145959596.
    assert regional.loc[regional.Zone == "NAZ", "60/20/20"].item() == "+0.11"
    report = {"status": "PASS", "input_checksums": len(manifest["sha256"]),
              "benchmark_cells": 52, "regional_cells": 40,
              "paper_rounding_difference": "NAZ 60/20/20: computed +0.11; original paper +0.12",
              "figure_mode": "Exact archived figures copied; use regenerate_plots.py for numerical rerenders."}
    (ROOT / "outputs/artifact-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
