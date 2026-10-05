#!/usr/bin/env python3
"""Create Table 2 from curated baseline benchmark CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "outputs"
RATIOS = ["50/25/25", "60/20/20", "70/15/15", "80/10/10"]

ORIGINAL_NAMES = {
    "USGS fixed P-LR / M1": "USGS",
    "Logistic regression": "LR",
    "CART": "CART",
    "SVM": "SVM",
    "Random forest": "RF",
    "HistGradientBoosting": "HGB",
    "NN": "NN",
}

AUG_NAMES = {
    "Logistic regression": "LR*",
    "CART": "CART*",
    "SVM": "SVM*",
    "Random forest": "RF*",
    "HistGradientBoosting": "HGB*",
    "NN+AUG selected": "NN*",
}


def fmt_cell(value: str) -> str:
    return "$" + value.replace(" +/- ", r" \pm ") + "$"


def load_rows() -> list[dict[str, str]]:
    orig = pd.read_csv(RESULTS / "baseline_original_clean_table.csv")
    aug = pd.read_csv(RESULTS / "baseline_augmented_clean_table.csv")
    rows: list[dict[str, str]] = []

    for source_name, display in ORIGINAL_NAMES.items():
        rec = orig[orig["Model"].eq(source_name)].iloc[0]
        rows.append({"Method": display, **{r: rec[r] for r in RATIOS}})

    for source_name, display in AUG_NAMES.items():
        rec = aug[aug["Model"].eq(source_name)].iloc[0]
        rows.append({"Method": display, **{r: rec[r] for r in RATIOS}})

    return rows


def write_tex(rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{50/25/25} & \textbf{60/20/20} & \textbf{70/15/15} & \textbf{80/10/10} \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if i in {1, 7}:
            lines.append(r"\midrule")
        method = row["Method"].replace("*", r"$^{*}$")
        cells = [method] + [fmt_cell(row[r]) for r in RATIOS]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table2_baseline_benchmark.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    df = pd.DataFrame(rows, columns=["Method"] + RATIOS)
    df.to_csv(OUT / "table2_baseline_benchmark.csv", index=False)
    write_tex(rows)
    print("Wrote outputs/table2_baseline_benchmark.csv")
    print("Wrote outputs/table2_baseline_benchmark.tex")


if __name__ == "__main__":
    main()
