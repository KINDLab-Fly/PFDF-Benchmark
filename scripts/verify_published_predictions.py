#!/usr/bin/env python3
"""Recompute Tables 2 and 3 from held-out predictions of all 640 historical checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RATIOS = ["50/25/25", "60/20/20", "70/15/15", "80/10/10"]
NAMES = {"FullLR": "Logistic regression", "CART": "CART", "SVM": "SVM",
         "RandomForest": "Random forest", "HistGradientBoosting": "HistGradientBoosting",
         "NN": "NN", "NN_AUG_SELECTED": "NN+AUG selected"}


def ts(y, scores):
    pred = scores >= .5
    tp = int(np.sum(pred & (y == 1)))
    denominator = tp + int(np.sum(pred & (y == 0))) + int(np.sum(~pred & (y == 1)))
    return tp / denominator if denominator else float("nan")


def finite_mean(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def main():
    directory = ROOT / "results/published_predictions"
    data = np.load(directory / "heldout-predictions.npz", allow_pickle=False)
    records = json.loads((directory / "prediction-records.json").read_text())["records"]
    rows = []
    for record in records:
        prefix = record["array_prefix"]
        indices, scores = data[prefix + "_indices"], data[prefix + "_scores"]
        score = ts(data["labels"][indices], scores)
        assert np.isclose(score, record["historical_TS"], rtol=0, atol=1e-12), prefix
        rows.append({**record, "TS": score})
    frame = pd.DataFrame(rows)
    means = frame.groupby(["config", "model", "ratio", "split_seed"]).TS.mean()
    comparisons = []
    for config in ["original", "augmented"]:
        source = pd.read_csv(ROOT / f"results/baseline_{config}_clean_table.csv").set_index("Model")
        for model, display in NAMES.items():
            if model not in means.loc[config].index.get_level_values("model"):
                continue
            for ratio in RATIOS:
                selected = model
                if model == "FullLR":
                    selected = max(["FullLR", "PenalizedLR"], key=lambda m: means.loc[config, m, ratio].mean())
                values = means.loc[config, selected, ratio].to_numpy()
                actual = [values.mean(), values.std()]
                expected = list(map(float, source.loc[display, ratio].split(" +/- ")))
                assert np.allclose(actual, expected, rtol=0, atol=.00005001), (config, display, ratio)
                comparisons.append({"config": config, "model": display, "ratio": ratio,
                                    "mean_TS": float(actual[0]), "std_TS": float(actual[1])})
    # Fixed USGS reference and region-by-region neural improvement use the same held-out indices.
    regional = []
    usgs = pd.read_csv(ROOT / "results/baseline_original_clean_table.csv").set_index("Model")
    for ratio in RATIOS:
        usgs_values = []
        for seed in range(42, 47):
            nn = [r for r in records if r["config"] == "original" and r["model"] == "NN"
                  and r["ratio"] == ratio and r["split_seed"] == seed]
            assert len(nn) == 10
            indices = data[nn[0]["array_prefix"] + "_indices"]
            labels, zones = data["labels"][indices], data["zones"][indices]
            usgs_scores = data["usgs_scores"][indices]
            usgs_values.append(ts(labels, usgs_scores))
            for zone in sorted(set(zones)):
                mask = zones == zone
                reference = ts(labels[mask], usgs_scores[mask])
                neural = finite_mean([ts(labels[mask], data[r["array_prefix"] + "_scores"][mask]) for r in nn])
                regional.append({"Zone": zone, "Ratio": ratio, "split_seed": seed,
                                 "Delta_TS_NN_minus_USGS": neural - reference})
        actual = [np.mean(usgs_values), np.std(usgs_values)]
        expected = list(map(float, usgs.loc["USGS fixed P-LR / M1", ratio].split(" +/- ")))
        assert np.allclose(actual, expected, rtol=0, atol=.00005001)
        comparisons.append({"config": "original", "model": "USGS", "ratio": ratio,
                            "mean_TS": float(actual[0]), "std_TS": float(actual[1])})
    region_means = pd.DataFrame(regional).groupby(["Zone", "Ratio"]).Delta_TS_NN_minus_USGS.mean()
    reference = pd.read_csv(ROOT / "results/region_performance/original_nn_minus_usgs_region_absolute_improvement_long_with_counts.csv")
    for row in reference.itertuples():
        actual = region_means.loc[row.Zone, row.Ratio]
        assert np.isclose(actual, row.Delta_TS_NN_minus_USGS, rtol=0, atol=1e-12, equal_nan=True), (row.Zone, row.Ratio, actual, row.Delta_TS_NN_minus_USGS)
    # Independent reconstruction of every feature-importance mean/std from all 50 learned policies.
    policies = np.load(directory / "policy-probabilities.npz", allow_pickle=False)
    policy_records = json.loads((directory / "policy-records.json").read_text())["records"]
    feature_rows = []
    for record in policy_records:
        for feature, probability in zip(record["feature_names"], policies[record["array_prefix"]]):
            feature_rows.append({"group_type": record["group_type"], "group_name": record["group_name"],
                                 "feature": feature, "probability": float(probability)})
    features = pd.DataFrame(feature_rows).groupby(["group_type", "group_name", "feature"]).probability.agg(["mean", "std"])
    feature_reference = pd.read_csv(ROOT / "results/rl_feature_selection_augmented_70-15-15_feature_mean.csv")
    for row in feature_reference.itertuples():
        actual = features.loc[row.group_type, row.group_name, row.feature]
        assert np.allclose(actual, [row.mean_remove_probability, row.std_remove_probability], rtol=0, atol=1e-12)
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    pd.DataFrame(comparisons).to_csv(out / "published-benchmark-from-predictions.csv", index=False)
    region_means.to_csv(out / "published-regions-from-predictions.csv")
    report = {"status": "PASS", "historical_checkpoint_metrics": len(records),
              "benchmark_cells": len(comparisons), "regional_cells": len(reference),
              "historical_rl_policies": len(policy_records), "feature_importance_rows": len(feature_reference),
              "mode": "Recomputed from historical checkpoint predictions, not copied from table CSVs or retrained."}
    (out / "published-prediction-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
