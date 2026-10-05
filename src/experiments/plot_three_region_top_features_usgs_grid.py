#!/usr/bin/env python3
"""Combine OSD, SWCO, and MT top-feature diagnostics into a 3x2 grid."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.reproduce_checkpointed_baselines import _load_complete_case  # noqa: E402


OUT_DIR = ROOT / "outputs/regenerated"
OUT_PNG = OUT_DIR / "three_region_top_features_usgs_grid.png"
OUT_PDF = OUT_DIR / "three_region_top_features_usgs_grid.pdf"
OUT_CSV = OUT_DIR / "three_region_top_features_usgs_grid_summary.csv"

COLORS = {
    "USGS inputs": "#4C78A8",
    "Top NN/RL features": "#D95F02",
    "No PFDF": "#4C78A8",
    "PFDF": "#D95F02",
}

REGION_CONFIGS = [
    {
        "region": "OSD",
        "left_title": "OSD",
        "right_title": "OSD",
        "features": [
            ("acc015", r"$R$", "USGS inputs"),
            ("x1", r"$X_1$", "USGS inputs"),
            ("x2", r"$X_2$", "USGS inputs"),
            ("x3", r"$X_3$", "USGS inputs"),
            ("storm_accum", "Storm\nAcc.", "Top NN/RL features"),
            ("acc030", "Pk30\nRain", "Top NN/RL features"),
            ("peak_i15", "Pk15\nInt.", "Top NN/RL features"),
            ("acc060", "Pk60\nRain", "Top NN/RL features"),
            ("storm_avg_i", "Avg.\nInt.", "Top NN/RL features"),
        ],
        "ylim_min": 3.6,
        "annotation_y": 0.94,
        "annotation_va": "top",
    },
    {
        "region": "SWCO",
        "left_title": "SWCO",
        "right_title": "SWCO",
        "features": [
            ("acc015", r"$R$", "USGS inputs"),
            ("x1", r"$X_1$", "USGS inputs"),
            ("x2", r"$X_2$", "USGS inputs"),
            ("x3", r"$X_3$", "USGS inputs"),
            ("gauge_dist", "Gauge\nDist.", "Top NN/RL features"),
            ("log_area", "Log Area", "Top NN/RL features"),
            ("soilm_30d", "SM3m", "Top NN/RL features"),
            ("storm_accum", "Storm\nAcc.", "Top NN/RL features"),
            ("slope_deg", "Slope", "Top NN/RL features"),
            ("storm_avg_i", "Avg.\nInt.", "Top NN/RL features"),
        ],
        "ylim_min": 2.25,
        "annotation_y": 0.94,
        "annotation_va": "top",
    },
    {
        "region": "MT",
        "left_title": "MT",
        "right_title": "MT",
        "features": [
            ("acc015", r"$R$", "USGS inputs"),
            ("x1", r"$X_1$", "USGS inputs"),
            ("x2", r"$X_2$", "USGS inputs"),
            ("x3", r"$X_3$", "USGS inputs"),
            ("slope_deg", "Slope", "Top NN/RL features"),
            ("acc030", "Pk30\nRain", "Top NN/RL features"),
            ("storm_accum", "Storm\nAcc.", "Top NN/RL features"),
            ("acc060", "Pk60\nRain", "Top NN/RL features"),
            ("peak_i15", "Pk15\nInt.", "Top NN/RL features"),
        ],
        "ylim_min": 1.75,
        "annotation_y": 0.08,
        "annotation_va": "bottom",
    },
]


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titleweight": "bold",
    })


def confusion(y_true: np.ndarray, scores: np.ndarray) -> dict[str, int | float]:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores, dtype=float) >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "recall": tp / (tp + fn) if (tp + fn) else np.nan,
    }


def ratio_value(no_mean: float, yes_mean: float) -> float:
    if not np.isfinite(no_mean) or not np.isfinite(yes_mean) or abs(no_mean) < 1e-12:
        return np.nan
    return yes_mean / no_mean


def build_summary(zdf: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for feature, label, group in config["features"]:
        no_mean = float(zdf.loc[zdf["pfdf"].eq(0), feature].mean())
        yes_mean = float(zdf.loc[zdf["pfdf"].eq(1), feature].mean())
        pooled_std = float(zdf[feature].std())
        rows.append({
            "region": config["region"],
            "feature": feature,
            "label": label,
            "group": group,
            "no_pfdf_mean": no_mean,
            "pfdf_mean": yes_mean,
            "pfdf_over_no_pfdf": ratio_value(no_mean, yes_mean),
            "standardized_difference": (yes_mean - no_mean) / pooled_std if pooled_std else np.nan,
        })
    cm = confusion(zdf["pfdf"].to_numpy(), zdf["usgs_p"].to_numpy())
    for cls, name in [(0, "No PFDF"), (1, "PFDF")]:
        vals = zdf.loc[zdf["pfdf"].eq(cls), "usgs_p"]
        rows.append({
            "region": config["region"],
            "feature": "usgs_p",
            "label": name,
            "group": "USGS probability",
            "no_pfdf_mean": float(vals.mean()) if cls == 0 else np.nan,
            "pfdf_mean": float(vals.mean()) if cls == 1 else np.nan,
            "pfdf_over_no_pfdf": np.nan,
            "standardized_difference": np.nan,
            **cm,
        })
    return pd.DataFrame(rows)


def plot_feature_ratios(ax, summary: pd.DataFrame, config: dict, show_legend: bool, show_ylabel: bool) -> None:
    feature_rows = summary[summary["feature"].ne("usgs_p")].copy()
    x = np.arange(len(feature_rows)) * 1.05
    bars = ax.bar(
        x,
        feature_rows["pfdf_over_no_pfdf"],
        width=0.64,
        color=[COLORS[group] for group in feature_rows["group"]],
        edgecolor="black",
        linewidth=0.9,
    )
    for bar, val in zip(bars, feature_rows["pfdf_over_no_pfdf"]):
        if not np.isfinite(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.035,
            f"{val:.2f}x",
            ha="center",
            va="bottom",
            fontsize=33,
            fontweight="bold",
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
    ax.set_xticks(x, feature_rows["label"])
    ax.tick_params(axis="x", labelsize=36, rotation=0, pad=7)
    for tick in ax.get_xticklabels():
        tick.set_ha("center")
        tick.set_multialignment("center")
    ax.tick_params(axis="y", labelsize=38)
    finite_vals = feature_rows["pfdf_over_no_pfdf"].replace([np.inf, -np.inf], np.nan).dropna()
    ymax = max(config["ylim_min"], float(finite_vals.max()) * 1.18)
    ax.set_ylim(0, ymax)
    if show_ylabel:
        ax.set_ylabel("PFDF / No-PFDF Mean", fontsize=46, fontweight="bold", labelpad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.42)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(config["region"], loc="left", fontsize=39, fontweight="bold", pad=4)


def plot_usgs_probability(
    ax,
    zdf: pd.DataFrame,
    config: dict,
    rng: np.random.Generator,
    show_legend: bool,
    show_ylabel: bool,
) -> None:
    for cls, name, x0 in [(0, "No PFDF", 0), (1, "PFDF", 1)]:
        vals = zdf.loc[zdf["pfdf"].eq(cls), "usgs_p"].to_numpy()
        jitter = rng.normal(0, 0.050, len(vals))
        ax.scatter(
            np.full(len(vals), x0) + jitter,
            vals,
            s=58,
            color=COLORS[name],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.82,
            zorder=3,
            label=name,
        )
        mean = float(vals.mean())
        ax.hlines(mean, x0 - 0.27, x0 + 0.27, color="black", linewidth=2.5)
        label_y = mean - 0.045 if mean > 0.90 else mean + 0.045
        label_va = "top" if mean > 0.90 else "bottom"
        ax.text(
            x0,
            label_y,
            f"{mean:.2f}",
            ha="center",
            va=label_va,
            fontsize=36,
            fontweight="bold",
        )

    cm = confusion(zdf["pfdf"].to_numpy(), zdf["usgs_p"].to_numpy())
    ax.text(
        0.5,
        config["annotation_y"],
        f"USGS misses {cm['FN']}/{cm['TP'] + cm['FN']} PFDF\n"
        f"false alarms {cm['FP']}/{cm['TN'] + cm['FP']}",
        transform=ax.transAxes,
        ha="center",
        va=config["annotation_va"],
        fontsize=38,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#999999"},
    )
    ax.axhline(0.5, color="#B2182B", linestyle="--", linewidth=1.8)

    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([0, 1], ["No PFDF", "PFDF"])
    ax.tick_params(axis="x", labelsize=39, rotation=0, pad=7)
    ax.tick_params(axis="y", labelsize=38)
    if show_ylabel:
        ax.set_ylabel("USGS predicted probability", fontsize=46, fontweight="bold", labelpad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.42)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(config["region"], loc="left", fontsize=39, fontweight="bold", pad=4)


def main() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    setup_style()
    df, _, _ = _load_complete_case()
    rng = np.random.default_rng(81)
    summaries = []

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(23.5, 17.4),
        gridspec_kw={"width_ratios": [1.34, 0.82], "hspace": 0.3, "wspace": 0.15},
    )
    for row, config in enumerate(REGION_CONFIGS):
        zdf = df[df["zone"].eq(config["region"])].copy()
        summary = build_summary(zdf, config)
        summaries.append(summary)
        plot_feature_ratios(axes[row, 0], summary, config, show_legend=(row == 0), show_ylabel=(row == 1))
        plot_usgs_probability(
            axes[row, 1],
            zdf,
            config,
            rng,
            show_legend=(row == 0),
            show_ylabel=(row == 1),
        )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["USGS inputs"], edgecolor="black"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["Top NN/RL features"], edgecolor="black"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["No PFDF"], markeredgecolor="black", markersize=22),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["PFDF"], markeredgecolor="black", markersize=22),
    ]
    fig.legend(
        legend_handles,
        ["USGS inputs", "Top NN/RL features", "No PFDF", "PFDF"],
        loc="upper center",
        ncol=4,
        frameon=False,
        facecolor="white",
        edgecolor="#BBBBBB",
        fontsize=39,
        bbox_to_anchor=(0.55, 0.995),
    )

    axes[2, 0].set_xlabel("Feature", fontsize=48, fontweight="bold", labelpad=10)
    axes[2, 1].set_xlabel("Observed class", fontsize=48, fontweight="bold", labelpad=10)
    fig.subplots_adjust(left=0.095, right=0.982, bottom=0.080, top=0.905, hspace=0.56, wspace=0.15)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(summaries, ignore_index=True).to_csv(OUT_CSV, index=False)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
