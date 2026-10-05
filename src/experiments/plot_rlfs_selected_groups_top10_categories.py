#!/usr/bin/env python3
"""Combined top-10 RL feature-importance bars for selected groups."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / Path("results/rl_feature_selection_augmented_70-15-15_feature_mean.csv")
OUT_DIR = ROOT / Path("outputs/regenerated/figures/rlfs_selected_groups")
OUT_PNG = OUT_DIR / "rl_feature_selection_selected_groups_top10_categories.png"
OUT_PDF = OUT_DIR / "rl_feature_selection_selected_groups_top10_categories.pdf"

GROUPS = [
    ("global", "global", "Global"),
    ("region", "OSD", "OSD"),
    ("region", "SGSBSJ", "SGSBSJ"),
    ("region", "SWCO", "SWCO"),
    ("region", "MT", "MT"),
]

CATEGORY = {
    "acc030": "Storm",
    "storm_accum": "Storm",
    "peak_i15": "Storm",
    "zone_ord": "Terrain",
    "storm_duration_signal": "Storm",
    "acc060": "Storm",
    "x3": "Soil",
    "x1": "Wildfire",
    "slope_deg": "Terrain",
    "storm_avg_i": "Storm",
    "log_area": "Terrain",
    "storm_dur": "Storm",
    "localized_extreme_signal": "Storm",
    "gauge_dist": "Terrain",
    "x2": "Wildfire",
    "soilm_30d": "Soil",
    "mentions_burn_scar_or_debris_flow": "Wildfire",
    "burn_scar_sensitivity_signal": "Wildfire",
    "precipitation_signal": "Storm",
    "antecedent_wetness_signal": "Storm",
    "overall_pfdf_weather_risk": "Storm",
    "soilm_7d": "Soil",
    "aspect_deg": "Terrain",
    "month_sin": "Storm",
    "month_cos": "Storm",
}

COLORS = {
    "Storm": "#A6CEE3",
    "Wildfire": "#F4A3A3",
    "Soil": "#C8A27A",
    "Terrain": "#B2DF8A",
}

HATCHES = {
    "Storm": "/",
    "Wildfire": "\\",
    "Soil": ".",
    "Terrain": "x",
}

LABELS = {
    "acc030": "Peak\n30-min\nRainfall",
    "storm_accum": "Storm\nAccumu-\nLation",
    "peak_i15": "Peak\n15-min\nIntensity",
    "zone_ord": "Geographic\nZone",
    "storm_duration_signal": "Forecast\nStorm\nPersistence",
    "acc060": "Peak\n60-min\nRainfall",
    "x3": "Soil\nErodibility",
    "x1": "Burn\nSeverity",
    "slope_deg": "Mean\nTerrain\nSlope",
    "storm_avg_i": "Average\nStorm\nIntensity",
    "log_area": "Contributing\nArea",
    "storm_dur": "Storm\nDuration",
    "localized_extreme_signal": "Localized\nExtreme\nRainfall",
    "gauge_dist": "Rain Gauge\nDistance",
    "x2": "Burn Severity\nMagnitude",
    "soilm_30d": "30-day\nSoil\nMoisture",
    "mentions_burn_scar_or_debris_flow": "Burn-scar\nDebris-flow\nMention",
    "burn_scar_sensitivity_signal": "Burn-scar\nSensitivity",
    "precipitation_signal": "Precipitation\nThreat",
    "antecedent_wetness_signal": "Antecedent\nWetness",
    "overall_pfdf_weather_risk": "Overall\nWeather\nRisk",
    "soilm_7d": "7-day\nSoil\nMoisture",
    "aspect_deg": "Mean\nTerrain\nAspect",
    "month_sin": "Seasonality\nsin(month)",
    "month_cos": "Seasonality\ncos(month)",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "hatch.linewidth": 0.45,
    })


def prepare_group(df: pd.DataFrame, group_type: str, group_name: str) -> pd.DataFrame:
    top = (
        df[df["group_type"].eq(group_type) & df["group_name"].eq(group_name)]
        .sort_values("mean_remove_probability", ascending=False)
        .head(10)
        .copy()
    )
    missing = sorted(set(top["feature"]) - set(CATEGORY))
    if missing:
        raise ValueError(f"Missing category mapping for features: {missing}")
    top["category"] = top["feature"].map(CATEGORY)
    top["label"] = top["feature"].map(LABELS).fillna(top["feature"])
    return top


def draw_group(ax, top: pd.DataFrame, title: str) -> None:
    x = np.arange(len(top)) * 2.15
    bars = ax.bar(
        x,
        top["mean_remove_probability"],
        width=0.55,
        color=[COLORS[c] for c in top["category"]],
        edgecolor="black",
        linewidth=1.0,
    )
    for bar, category in zip(bars, top["category"]):
        bar.set_hatch(HATCHES[category])

    ax.set_title(title, fontsize=50, fontweight="bold", loc="left")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x, top["label"])
    ax.set_xlim(x[0] - 0.9, x[-1] + 0.9)
    ax.tick_params(axis="x", labelsize=53, rotation=30)
    ax.tick_params(axis="y", labelsize=41)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
        tick.set_multialignment("center")
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def main() -> None:
    setup_style()
    plt.rcParams["pdf.fonttype"] = 42
    df = pd.read_csv(INPUT)

    fig, axes = plt.subplots(len(GROUPS), 1, figsize=(30, 36), sharey=True)
    for ax, (group_type, group_name, title) in zip(axes, GROUPS):
        draw_group(ax, prepare_group(df, group_type, group_name), title)
        ax.set_ylabel("Feature\nImportance", fontsize=49, fontweight="bold")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[name], edgecolor="black", hatch=HATCHES[name])
        for name in ["Storm", "Wildfire", "Soil", "Terrain"]
    ]
    fig.legend(
        handles,
        ["Storm", "Wildfire", "Soil", "Terrain"],
        loc="upper center",
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#B0B0B0",
        fontsize=50,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=100, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
