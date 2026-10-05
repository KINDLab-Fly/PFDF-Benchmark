#!/usr/bin/env python3
"""Compute regionwise TS improvement from USGS fixed LR to original-feature NN."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.eval_literature_baselines import _build_feature_matrix  # noqa: E402
from experiments.reproduce_checkpointed_baselines import _load_complete_case  # noqa: E402
from features.split import zone_stratified_split_3way  # noqa: E402
from models.mlp import EmbeddingMLP  # noqa: E402
from training.hp_search import EVAL_SEEDS, SPLIT_RATIOS  # noqa: E402


OUT_DIR = ROOT / "outputs/refit-region-performance"
RECORDS_PATH = ROOT / "results/checkpointed_baseline_original_checkpoint_records.json"
SPLIT_SEEDS = [42, 43, 44, 45, 46]


def threat_score(y_true, scores_or_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores_or_pred, dtype=float) >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")


def load_nn_scores(checkpoint_path: str | Path, X: np.ndarray) -> np.ndarray:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hp = ckpt["metadata"]["best_hp"]
    model = EmbeddingMLP(X.shape[1], hp["hidden"], hp["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, paired, _ = _load_complete_case()
    y_all = df["pfdf"].values.astype(int)
    zones = df["zone"].values
    usgs_scores = df["usgs_p"].values.astype(float)

    with RECORDS_PATH.open() as f:
        records = json.load(f)
    nn_records = {
        (rec["ratio_label"], int(rec["split_seed"]), int(rec["eval_seed"])): rec
        for rec in records
        if rec["model"] == "NN"
    }

    rows = []
    for train_ratio, val_ratio, ratio_label in SPLIT_RATIOS:
        X = None
        for split_seed in SPLIT_SEEDS:
            tr, va, te = zone_stratified_split_3way(
                paired, train_ratio=train_ratio, val_ratio=val_ratio, seed=split_seed
            )
            if X is None:
                X = _build_feature_matrix("n", df, tr, None)
            else:
                X = _build_feature_matrix("n", df, tr, None)

            test_zones = sorted(set(zones[te]))
            nn_scores_by_seed = {}
            for eval_seed in EVAL_SEEDS:
                rec = nn_records[(ratio_label, split_seed, eval_seed)]
                ckpt_path = Path(rec["checkpoint"]) if Path(rec["checkpoint"]).is_absolute() else ROOT / rec["checkpoint"]
                nn_scores_by_seed[eval_seed] = load_nn_scores(ckpt_path, X)

            for zone in test_zones:
                idx = np.asarray(te)[zones[te] == zone]
                if len(idx) == 0:
                    continue
                usgs_ts = threat_score(y_all[idx], usgs_scores[idx])
                nn_ts_values = [
                    threat_score(y_all[idx], nn_scores_by_seed[eval_seed][idx])
                    for eval_seed in EVAL_SEEDS
                ]
                nn_ts_values = [v for v in nn_ts_values if math.isfinite(v)]
                nn_ts = float(np.mean(nn_ts_values)) if nn_ts_values else float("nan")
                improvement = nn_ts - usgs_ts if math.isfinite(usgs_ts) and math.isfinite(nn_ts) else float("nan")
                rows.append({
                    "zone": zone,
                    "ratio": ratio_label,
                    "split_seed": split_seed,
                    "n_test": int(len(idx)),
                    "n_positive": int(y_all[idx].sum()),
                    "usgs_ts": usgs_ts,
                    "nn_ts_mean_eval_seeds": nn_ts,
                    "improvement_nn_minus_usgs": improvement,
                })

    long_df = pd.DataFrame(rows)
    long_path = OUT_DIR / "original_nn_minus_usgs_region_improvement_long.csv"
    long_df.to_csv(long_path, index=False)

    summary = (
        long_df.groupby(["zone", "ratio"], as_index=False)
        .agg(
            mean_n_test=("n_test", "mean"),
            mean_n_positive=("n_positive", "mean"),
            usgs_ts_mean=("usgs_ts", "mean"),
            nn_ts_mean=("nn_ts_mean_eval_seeds", "mean"),
            improvement_mean=("improvement_nn_minus_usgs", "mean"),
            improvement_std=("improvement_nn_minus_usgs", "std"),
            n_split_seeds=("improvement_nn_minus_usgs", "count"),
        )
    )
    summary_path = OUT_DIR / "original_nn_minus_usgs_region_improvement_summary.csv"
    summary.to_csv(summary_path, index=False)

    improvement_wide = summary.pivot(index="zone", columns="ratio", values="improvement_mean")
    improvement_wide = improvement_wide[[r[2] for r in SPLIT_RATIOS]]
    improvement_wide_path = OUT_DIR / "original_nn_minus_usgs_region_improvement_wide.csv"
    improvement_wide.to_csv(improvement_wide_path)

    markdown = improvement_wide.applymap(lambda x: "" if pd.isna(x) else f"{x:+.3f}")
    markdown_path = OUT_DIR / "original_nn_minus_usgs_region_improvement_wide.md"
    markdown_path.write_text(markdown.reset_index().to_markdown(index=False) + "\n")

    print(f"Saved {long_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {improvement_wide_path}")
    print(markdown.reset_index().to_markdown(index=False))


if __name__ == "__main__":
    main()
