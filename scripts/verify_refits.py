#!/usr/bin/env python3
"""Compare actual retraining results with all saved Table 2 mean/std entries."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiments.reproduce_checkpointed_baselines import _load_complete_case, _metrics
from features.split import zone_stratified_split_3way
from training.hp_search import SPLIT_RATIOS

NAMES = {"FullLR": "Logistic regression", "CART": "CART", "SVM": "SVM",
         "RandomForest": "Random forest", "HistGradientBoosting": "HistGradientBoosting",
         "NN": "NN", "NN_AUG_SELECTED": "NN+AUG selected"}


def main():
    comparisons = []
    for config in ["original", "augmented"]:
        records = pd.DataFrame(json.loads((ROOT / f"results/checkpointed_baseline_{config}_checkpoint_records.json").read_text()))
        records["TS"] = records.test_metrics.map(lambda x: x["TS"])
        source = pd.read_csv(ROOT / f"results/baseline_{config}_clean_table.csv").set_index("Model")
        means = records.groupby(["model", "ratio_label", "split_seed"]).TS.mean()
        for model, display in NAMES.items():
            if model not in means.index.get_level_values("model"):
                continue
            for _, _, ratio in SPLIT_RATIOS:
                selected_model = model
                if model == "FullLR":
                    selected_model = max(["FullLR", "PenalizedLR"], key=lambda m: means.loc[m, ratio].mean())
                values = means.loc[selected_model, ratio].to_numpy()
                assert len(values) == 5, (config, model, ratio)
                expected = list(map(float, source.loc[display, ratio].split(" +/- ")))
                observed = [float(values.mean()), float(values.std())]
                comparisons.append({"config": config, "model": selected_model, "ratio": ratio,
                                    "observed_mean": observed[0], "observed_std": observed[1],
                                    "saved_mean": expected[0], "saved_std": expected[1],
                                    "matches_saved_4_decimals": bool(np.allclose(observed, expected, rtol=0, atol=.00005001))})
    # Recompute the operational reference on all 20 canonical test splits too.
    df, paired, _ = _load_complete_case()
    source = pd.read_csv(ROOT / "results/baseline_original_clean_table.csv").set_index("Model")
    for train, val, ratio in SPLIT_RATIOS:
        scores = []
        for seed in range(42, 47):
            _, _, te = zone_stratified_split_3way(paired, train, val, seed)
            scores.append(_metrics(df.pfdf.to_numpy()[te], df.usgs_p.to_numpy()[te])["TS"])
        observed = [float(np.mean(scores)), float(np.std(scores))]
        expected = list(map(float, source.loc["USGS fixed P-LR / M1", ratio].split(" +/- ")))
        comparisons.append({"config": "original", "model": "USGS", "ratio": ratio,
                            "observed_mean": observed[0], "observed_std": observed[1],
                            "saved_mean": expected[0], "saved_std": expected[1],
                            "matches_saved_4_decimals": bool(np.allclose(observed, expected, rtol=0, atol=.00005001))})
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    pd.DataFrame(comparisons).to_csv(out / "refit-comparison.csv", index=False)
    mismatches = [x for x in comparisons if not x["matches_saved_4_decimals"]]
    report = {"cells_checked": len(comparisons), "cells_matching_saved_4_decimals": len(comparisons)-len(mismatches),
              "mismatches": mismatches,
              "scope": "Actual saved-HP model refits, 4 ratios x 5 split seeds; 10 neural initialization seeds. No new HP search."}
    (out / "refit-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
