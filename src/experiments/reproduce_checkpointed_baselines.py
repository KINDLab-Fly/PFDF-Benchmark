"""
Reproduce original/selected-augmented baseline tables with checkpoints.

This script uses the saved best hyperparameters from the completed campaigns,
refits the corresponding predictive model on each canonical split, evaluates
the held-out test fold, and saves the fitted model objects.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.exceptions import ConvergenceWarning

os.chdir(Path(__file__).resolve().parents[2])
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.eval_literature_baselines import _build_feature_matrix, _model_grid
from features.build import CORE_FEATURES, N_ALL_COLS, build_dataframe
from features.load import load_data
from features.split import zone_stratified_split_3way
from features.weather_structured import load_weather_structured_raw
from models.mlp import EmbeddingMLP
from training.hp_search import EVAL_SEEDS, MAX_EPOCHS, PATIENCE, SPLIT_RATIOS
from training.metrics import set_seed


CLASSICAL_MODELS = ["FullLR", "PenalizedLR", "CART", "SVM", "RandomForest", "HistGradientBoosting"]
DEFAULT_SPLIT_SEEDS = [42, 43, 44, 45, 46]

FEATURE_CONFIGS = {
    "original": {
        "feature_set": "n",
        "checkpoint_model_label": "NN",
        "nn_result_file": Path("results/core_benchmark_campaign_n.json"),
        "nn_result_key": "N",
        "classical_files": [
            Path("results/literature_baseline_campaign_literature-linear.json"),
            Path("results/literature_baseline_campaign_literature-svm.json"),
            Path("results/literature_baseline_campaign_literature-tree.json"),
        ],
        "out_prefix": "checkpointed_baseline_original",
    },
    "augmented": {
        "feature_set": "selected_aug",
        "checkpoint_model_label": "NN_AUG_SELECTED",
        "nn_result_file": Path("results/n_targeted_combo_eval_n-targeted-combo.json"),
        "nn_result_key": "N+TargetedCombo",
        "classical_files": [
            Path("results/literature_baseline_campaign_literature-aug-linear.json"),
            Path("results/literature_baseline_campaign_literature-aug-svm.json"),
            Path("results/literature_baseline_campaign_literature-aug-tree.json"),
        ],
        "out_prefix": "checkpointed_baseline_augmented",
    },
}

DISPLAY_NAMES = {
    "FullLR": "Full LR",
    "PenalizedLR": "Penalized LR",
    "CART": "CART",
    "SVM": "SVM",
    "RandomForest": "Random forest",
    "HistGradientBoosting": "HistGradientBoosting",
    "NN": "NN",
    "NN_AUG_SELECTED": "NN+AUG selected",
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _save_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _threat_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")


def _metrics(y_true, scores) -> dict:
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= 0.5).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return {
        "TS": _threat_score(y_true, pred),
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else float("nan"),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def _score_model(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return 1.0 / (1.0 + np.exp(-model.decision_function(X)))
    return model.predict(X).astype(float)


def _load_complete_case():
    events = load_data()
    with open("data/augmented_features.json") as f:
        aug_raw = json.load(f)
    df_full = build_dataframe(events, aug_raw)
    complete_mask = df_full[CORE_FEATURES].notna().all(axis=1)
    orig_idx = df_full.index[complete_mask].tolist()
    ev_cc = [events[i] for i in orig_idx]
    aug_cc = {str(new_i): aug_raw.get(str(old_i), {}) for new_i, old_i in enumerate(orig_idx)}
    df = build_dataframe(ev_cc, aug_cc)
    paired = [(ev, aug_cc.get(str(i), {})) for i, ev in enumerate(ev_cc)]
    w_struct_raw, _ = load_weather_structured_raw(ev_cc)
    return df, paired, w_struct_raw


def _classical_record(config_name: str, model_name: str, ratio_label: str, split_seed: int) -> dict:
    for path in FEATURE_CONFIGS[config_name]["classical_files"]:
        data = _load_json(path)
        if ratio_label in data and model_name in data[ratio_label]:
            for rec in data[ratio_label][model_name]["split_seed_results"]:
                if int(rec["split_seed"]) == int(split_seed):
                    return rec
    raise KeyError(f"Missing classical record for {config_name} {model_name} {ratio_label} seed={split_seed}")


def _nn_record(config_name: str, ratio_label: str, split_seed: int) -> dict:
    cfg = FEATURE_CONFIGS[config_name]
    data = _load_json(cfg["nn_result_file"])
    for rec in data[ratio_label][cfg["nn_result_key"]]["split_seed_results"]:
        if int(rec["split_seed"]) == int(split_seed):
            return rec
    raise KeyError(f"Missing NN record for {config_name} {ratio_label} seed={split_seed}")


def _classical_from_hp(model_name: str, hp: dict, split_seed: int):
    for candidate_hp, model in _model_grid(model_name, split_seed):
        if candidate_hp == hp:
            return model
    raise KeyError(f"Could not reconstruct {model_name} from hp={hp}")


def _train_nn_model(hp: dict, eval_seed: int, X_tr, y_tr, X_va, y_va, X_te, y_te):
    set_seed(eval_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingMLP(X_tr.shape[1], hp["hidden"], hp["dropout"]).to(device)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_te, dtype=torch.float32, device=device)

    n_pos = max(float(y_tr_t.sum()), 1.0)
    n_neg = float(len(y_tr_t)) - n_pos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp.get("weight_decay", 0.0))

    best_val = -1.0
    best_test = float("nan")
    best_state = None
    no_imp = 0
    for _ in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        crit(model(X_tr_t), y_tr_t).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = (torch.sigmoid(model(X_va_t)) >= 0.5).long().cpu().numpy()
            test_pred = (torch.sigmoid(model(X_te_t)) >= 0.5).long().cpu().numpy()
        val_ts = _threat_score(y_va, val_pred)
        test_ts = _threat_score(y_te, test_pred)
        if val_ts > best_val:
            best_val = val_ts
            best_test = test_ts
            best_state = copy.deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(X_te_t)).cpu().numpy()
    return model.cpu(), best_val, best_test, scores


def _checkpoint_dir(config_name: str, ratio_label: str, split_seed: int) -> Path:
    return Path("results/checkpoints/baselines") / config_name / _slug(ratio_label) / f"split_seed_{split_seed}"


def reproduce_config(config_name: str, ratios: list[str] | None, split_seeds: list[int], eval_seeds: list[int], models: list[str] | None):
    cfg = FEATURE_CONFIGS[config_name]
    selected_splits = [x for x in SPLIT_RATIOS if ratios is None or x[2] in ratios]
    selected_models = models or CLASSICAL_MODELS + [cfg["checkpoint_model_label"]]

    valid_models = set(CLASSICAL_MODELS + [cfg["checkpoint_model_label"]])
    if not selected_splits or (ratios and set(ratios) - {r[2] for r in SPLIT_RATIOS}):
        raise ValueError("Select valid ratio labels from SPLIT_RATIOS")
    if set(selected_models) - valid_models:
        raise ValueError(f"Invalid models for {config_name}: {set(selected_models) - valid_models}")
    if not split_seeds or not eval_seeds:
        raise ValueError("At least one split seed and evaluation seed are required")

    df, paired, w_struct_raw = _load_complete_case()
    y_all = df["pfdf"].values.astype(int)
    all_records = []

    for train_ratio, val_ratio, ratio_label in selected_splits:
        for split_seed in split_seeds:
            tr, va, te = zone_stratified_split_3way(paired, train_ratio=train_ratio, val_ratio=val_ratio, seed=split_seed)
            X = _build_feature_matrix(cfg["feature_set"], df, tr, w_struct_raw)
            if cfg["feature_set"] == "n":
                feature_names = N_ALL_COLS
            else:
                feature_names = N_ALL_COLS + ["slope_deg", "soilm_30d"] + [
                    "storm_duration_signal",
                    "antecedent_wetness_signal",
                    "burn_scar_sensitivity_signal",
                    "mentions_burn_scar_or_debris_flow",
                    "overall_pfdf_weather_risk",
                    "precipitation_signal",
                    "localized_extreme_signal",
                ]

            ckpt_dir = _checkpoint_dir(config_name, ratio_label, split_seed)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            base_meta = {
                "config_name": config_name,
                "feature_set": cfg["feature_set"],
                "ratio_label": ratio_label,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "split_seed": split_seed,
                "train_idx": list(map(int, tr)),
                "val_idx": list(map(int, va)),
                "test_idx": list(map(int, te)),
                "feature_names": feature_names,
            }

            for model_name in selected_models:
                print(f"[{config_name}] {ratio_label} split_seed={split_seed} model={model_name}", flush=True)
                if model_name in CLASSICAL_MODELS:
                    rec = _classical_record(config_name, model_name, ratio_label, split_seed)
                    hp = rec["best_hp"]
                    model = _classical_from_hp(model_name, hp, split_seed)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        model.fit(X[tr], y_all[tr])
                    scores = _score_model(model, X[te])
                    metrics = _metrics(y_all[te], scores)
                    ckpt_path = ckpt_dir / f"{model_name}.pkl"
                    _save_pickle(ckpt_path, {"model": model, "metadata": {**base_meta, "model_name": model_name, "best_hp": hp}})
                    all_records.append({
                        **base_meta,
                        "model": model_name,
                        "display_model": DISPLAY_NAMES[model_name],
                        "checkpoint": str(ckpt_path),
                        "best_hp": hp,
                        "eval_seed": None,
                        "test_metrics": metrics,
                    })
                elif model_name == cfg["checkpoint_model_label"]:
                    rec = _nn_record(config_name, ratio_label, split_seed)
                    hp = rec["best_hp"]
                    test_ts = []
                    for eval_seed in eval_seeds:
                        nn, val_ts, ts, scores = _train_nn_model(
                            hp, eval_seed, X[tr], y_all[tr], X[va], y_all[va], X[te], y_all[te]
                        )
                        ckpt_path = ckpt_dir / f"{model_name}_eval_seed_{eval_seed}.pt"
                        torch.save({
                            "state_dict": nn.state_dict(),
                            "metadata": {**base_meta, "model_name": model_name, "best_hp": hp, "eval_seed": eval_seed},
                        }, ckpt_path)
                        metrics = _metrics(y_all[te], scores)
                        metrics["TS"] = float(ts)
                        test_ts.append(float(ts))
                        all_records.append({
                            **base_meta,
                            "model": model_name,
                            "display_model": DISPLAY_NAMES[model_name],
                            "checkpoint": str(ckpt_path),
                            "best_hp": hp,
                            "eval_seed": eval_seed,
                            "val_TS": float(val_ts),
                            "test_metrics": metrics,
                        })
                    print(f"  NN eval mean TS={np.mean(test_ts):.4f} std={np.std(test_ts):.4f}", flush=True)
                else:
                    raise ValueError(f"Unknown requested model {model_name!r} for config {config_name}")

    records_path = Path("results") / f"{cfg['out_prefix']}_checkpoint_records.json"
    _save_json(records_path, all_records)
    _write_summary_tables(config_name, all_records)
    print(f"Saved checkpoint records -> {records_path}")


def _write_summary_tables(config_name: str, records: list[dict]) -> None:
    cfg = FEATURE_CONFIGS[config_name]
    rows = []
    for model_name in CLASSICAL_MODELS + [cfg["checkpoint_model_label"]]:
        model_records = [r for r in records if r["model"] == model_name]
        if not model_records:
            continue
        for _, _, ratio_label in SPLIT_RATIOS:
            ratio_records = [r for r in model_records if r["ratio_label"] == ratio_label]
            if not ratio_records:
                continue
            if model_name == cfg["checkpoint_model_label"]:
                per_split = []
                for split_seed in sorted({r["split_seed"] for r in ratio_records}):
                    vals = [r["test_metrics"]["TS"] for r in ratio_records if r["split_seed"] == split_seed]
                    per_split.append(float(np.mean(vals)))
                mean = float(np.mean(per_split))
                std = float(np.std(per_split))
            else:
                vals = [r["test_metrics"]["TS"] for r in ratio_records]
                mean = float(np.mean(vals))
                std = float(np.std(vals))
            rows.append({
                "config": config_name,
                "model": model_name,
                "display_model": DISPLAY_NAMES[model_name],
                "ratio": ratio_label,
                "mean_TS": mean,
                "std_TS": std,
                "n_records": len(ratio_records),
            })

    long_df = pd.DataFrame(rows)
    long_path = Path("results") / f"{cfg['out_prefix']}_checkpoint_summary_long.csv"
    long_df.to_csv(long_path, index=False)

    clean_labels = {
        "original": ["Logistic regression", "CART", "SVM", "Random forest", "HistGradientBoosting", "NN"],
        "augmented": ["Logistic regression", "CART", "SVM", "Random forest", "HistGradientBoosting", "NN+AUG selected"],
    }[config_name]
    table_rows = []
    for label in clean_labels:
        row = {"Model": label}
        for _, _, ratio_label in SPLIT_RATIOS:
            if label == "Logistic regression":
                cands = long_df[(long_df["ratio"] == ratio_label) & (long_df["model"].isin(["FullLR", "PenalizedLR"]))]
                if len(cands):
                    item = cands.sort_values("mean_TS", ascending=False).iloc[0]
                    row[ratio_label] = f"{item.mean_TS:.4f} +/- {item.std_TS:.4f}"
                else:
                    row[ratio_label] = ""
            else:
                item = long_df[(long_df["ratio"] == ratio_label) & (long_df["display_model"] == label)]
                if len(item):
                    item = item.iloc[0]
                    row[ratio_label] = f"{item.mean_TS:.4f} +/- {item.std_TS:.4f}"
                else:
                    row[ratio_label] = ""
        table_rows.append(row)
    wide_df = pd.DataFrame(table_rows)
    wide_csv = Path("results") / f"{cfg['out_prefix']}_table.csv"
    wide_md = Path("results") / f"{cfg['out_prefix']}_table.md"
    wide_df.to_csv(wide_csv, index=False)
    wide_md.write_text(wide_df.to_markdown(index=False) + "\n")
    print(wide_df.to_markdown(index=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=FEATURE_CONFIGS.keys(), required=True)
    parser.add_argument("--ratios", nargs="+", default=None, help="Optional ratio labels, e.g. 70/15/15")
    parser.add_argument("--split-seeds", nargs="+", type=int, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=EVAL_SEEDS)
    parser.add_argument("--models", nargs="+", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reproduce_config(args.config, args.ratios, args.split_seeds, args.eval_seeds, args.models)
