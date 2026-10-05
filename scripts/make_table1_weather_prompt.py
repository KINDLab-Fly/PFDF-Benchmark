#!/usr/bin/env python3
"""Create Table 1: the AFD-to-structured-weather example."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

SUMMARY = (
    "Two storm systems are expected with moderate to significant rainfall, "
    "especially in foothills and mountains due to orographic enhancement."
)

FEATURES = [
    ("precipitation_signal", 1),
    ("localized_extreme_signal", 0),
    ("short_duration_intensity_signal", 0),
    ("mentions_heavy_rain", 0),
    ("storm_duration_signal", 1),
    ("mentions_thunderstorm", 0),
    ("training_repetition_signal", 0),
    ("mentions_flash_flood", 0),
    ("antecedent_wetness_signal", 0),
    ("mentions_burn_scar_or_debris_flow", 0),
    ("convective_signal", 0),
    ("mentions_training_or_repeating_cells", 0),
    ("orographic_enhancement_signal", 1),
    ("mentions_slow_storm_motion", 0),
    ("explicit_flash_flood_signal", 0),
    ("mentions_monsoon_or_tropical_moisture", 0),
    ("burn_scar_sensitivity_signal", 0),
    ("overall_pfdf_weather_risk", 1),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    md = [
        "# Table 1: AFD Structured Weather Example",
        "",
        "**Input AFD:** `AFDLOX@2009-12-09T22:01`",
        "",
        "**Summary:** " + SUMMARY,
        "",
        "| Feature | Value |",
        "|---|---:|",
    ]
    md += [f"| `{name}` | {value} |" for name, value in FEATURES]
    (OUT / "table1_weather_prompt.md").write_text("\n".join(md) + "\n")

    rows = []
    for left, right in zip(FEATURES[::2], FEATURES[1::2]):
        rows.append(
            rf"\texttt{{{left[0].replace('_', chr(92) + '_')}}}={left[1]} & "
            rf"\texttt{{{right[0].replace('_', chr(92) + '_')}}}={right[1]} \\"
        )
    tex = "\n".join([
        r"\begin{tabular}{ll}",
        *rows,
        r"\end{tabular}",
    ])
    (OUT / "table1_weather_prompt.tex").write_text(tex + "\n")
    print("Wrote outputs/table1_weather_prompt.md")
    print("Wrote outputs/table1_weather_prompt.tex")


if __name__ == "__main__":
    main()
