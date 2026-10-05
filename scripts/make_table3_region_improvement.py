#!/usr/bin/env python3
"""Create Table 3 from curated region-wise NN-minus-USGS results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/region_performance"
OUT = ROOT / "outputs"
RATIOS = ["50/25/25", "60/20/20", "70/15/15", "80/10/10"]
ZONE_ORDER = ["OSD", "SGSBSJ", "VEN", "NAZ", "SAZ", "CNM", "FRCO", "SWCO", "WCO", "MT"]

AREA = {
    "OSD": "Southern California",
    "SGSBSJ": "Southern California",
    "VEN": "Southern California",
    "NAZ": "Arizona",
    "SAZ": "Arizona",
    "CNM": "N-Mexico",
    "FRCO": "Colorado / Western Plateau",
    "SWCO": "Colorado / Western Plateau",
    "WCO": "Colorado / Western Plateau",
    "MT": "Northern Rockies",
}


def fmt_delta(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:+.2f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    long = pd.read_csv(RESULTS / "original_nn_minus_usgs_region_absolute_improvement_long_with_counts.csv")
    wide = long.pivot(index="Zone", columns="Ratio", values="Delta_TS_NN_minus_USGS")
    records = long.groupby("Zone")["region_records_pos_neg"].first()

    rows = []
    for zone in ZONE_ORDER:
        row = {
            "Area": AREA[zone],
            "Zone": zone,
            "Records": records.get(zone, ""),
        }
        for ratio in RATIOS:
            row[ratio] = fmt_delta(wide.loc[zone, ratio]) if zone in wide.index and ratio in wide.columns else "N/A"
        rows.append(row)

    df = pd.DataFrame(rows, columns=["Area", "Zone", "Records"] + RATIOS)
    df.to_csv(OUT / "table3_region_improvement.csv", index=False)

    lines = [
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        r"\textbf{Area} & \textbf{Zone} & \textbf{Records} & \textbf{50/25/25} & \textbf{60/20/20} & \textbf{70/15/15} & \textbf{80/10/10} \\",
        r"\midrule",
    ]
    last_area = None
    for row in rows:
        area = row["Area"] if row["Area"] != last_area else ""
        if last_area is not None and row["Area"] != last_area:
            lines.append(r"\midrule")
        last_area = row["Area"]
        cells = [area, row["Zone"], row["Records"]] + [row[r] for r in RATIOS]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table3_region_improvement.tex").write_text("\n".join(lines) + "\n")

    print("Wrote outputs/table3_region_improvement.csv")
    print("Wrote outputs/table3_region_improvement.tex")


if __name__ == "__main__":
    main()
