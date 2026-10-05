"""
experiments/eval_literature_baselines.py — Classical literature baselines.

This runner evaluates non-neural baselines on exactly the same complete-case
data, split rule, split ratios, split seeds, and feature tensor used by the
canonical N baseline.  It intentionally excludes R*, graph, augmented,
weather, and USGS-probability input features.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

os.chdir(Path(__file__).resolve().parents[2])
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.logging_utils import create_run_logger
from features.build import CORE_FEATURES, build_dataframe, build_n_tensor
from features.load import load_data
from features.split import zone_stratified_split_3way
from features.weather_structured import STRUCTW_COLS, load_weather_structured_raw
from training.hp_search import SPLIT_RATIOS

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


MODEL_ORDER = [
    "FullLR",
    "PenalizedLR",
    "CART",
    "SVM",
    "RandomForest",
    "HistGradientBoosting",
]
DEFAULT_SPLIT_SEEDS = [42, 43, 44, 45, 46]

SELECTED_STRUCTW_COLS = [
    "storm_duration_signal",
    "antecedent_wetness_signal",
    "burn_scar_sensitivity_signal",
    "mentions_burn_scar_or_debris_flow",
    "overall_pfdf_weather_risk",
    "precipitation_signal",
    "localized_extreme_signal",
]

SELECTED_AUG_COLS = [
    "slope_deg",
    "soilm_30d",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _threat_score_from_labels(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    denom = tp + fp + fn
    return tp / denom if denom else float("nan")


def _metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else float("nan")
    out = {
        "TS": _threat_score_from_labels(y_true, y_pred),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["AUC"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["AUC"] = None
    else:
        out["AUC"] = None
    return out


def _score_and_pred(model, X):
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
        pred = (score >= 0.5).astype(int)
        return pred, score
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        pred = (score >= 0.0).astype(int)
        return pred, score
    pred = model.predict(X).astype(int)
    return pred, None


def _aggregate_model_ratio(records: list[dict]) -> dict:
    scores = [r["test_metrics"]["TS"] for r in records]
    return {
        "n_split_seeds": len(records),
        "split_seed_results": records,
        "mean_over_split_seeds": float(np.mean(scores)) if scores else float("nan"),
        "std_over_split_seeds": float(np.std(scores)) if scores else float("nan"),
    }


def _build_progress_snapshot(started_at, run_id, models, split_ratios, split_seeds, completed):
    total = len(models) * len(split_ratios) * len(split_seeds)
    done = len(completed)
    elapsed = time.time() - started_at
    rate = elapsed / done if done else None
    return {
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "total_tasks": total,
        "completed_tasks": done,
        "remaining_tasks": total - done,
        "elapsed_sec": elapsed,
        "eta_sec": rate * (total - done) if rate is not None else None,
        "latest_result": completed[-1] if completed else None,
        "models": models,
        "split_ratios": split_ratios,
        "split_seeds": split_seeds,
    }


def _model_grid(model_name: str, split_seed: int):
    if model_name == "FullLR":
        return [
            (
                {"class_weight": cw},
                LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    class_weight=cw,
                    max_iter=5000,
                    random_state=split_seed,
                ),
            )
            for cw in [None, "balanced"]
        ]

    if model_name == "PenalizedLR":
        grid = []
        for cw in [None, "balanced"]:
            for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
                grid.append((
                    {"penalty": "l2", "C": c, "class_weight": cw},
                    LogisticRegression(
                        penalty="l2",
                        C=c,
                        solver="lbfgs",
                        class_weight=cw,
                        max_iter=5000,
                        random_state=split_seed,
                    ),
                ))
                grid.append((
                    {"penalty": "l1", "C": c, "class_weight": cw},
                    LogisticRegression(
                        penalty="l1",
                        C=c,
                        solver="liblinear",
                        class_weight=cw,
                        max_iter=5000,
                        random_state=split_seed,
                    ),
                ))
            for c in [0.01, 0.1, 1.0, 10.0]:
                for l1_ratio in [0.25, 0.5, 0.75]:
                    grid.append((
                        {
                            "penalty": "elasticnet",
                            "C": c,
                            "l1_ratio": l1_ratio,
                            "class_weight": cw,
                        },
                        LogisticRegression(
                            penalty="elasticnet",
                            C=c,
                            l1_ratio=l1_ratio,
                            solver="saga",
                            class_weight=cw,
                            max_iter=5000,
                            random_state=split_seed,
                        ),
                    ))
        return grid

    if model_name == "CART":
        return [
            (
                {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_leaf,
                    "class_weight": cw,
                },
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    class_weight=cw,
                    random_state=split_seed,
                ),
            )
            for cw in [None, "balanced"]
            for max_depth in [2, 3, 4, 5, None]
            for min_leaf in [1, 5, 10, 20]
        ]

    if model_name == "SVM":
        return [
            (
                {
                    "kernel": "rbf",
                    "C": c,
                    "gamma": gamma,
                    "class_weight": cw,
                },
                SVC(
                    kernel="rbf",
                    C=c,
                    gamma=gamma,
                    class_weight=cw,
                    probability=True,
                    random_state=split_seed,
                ),
            )
            for cw in [None, "balanced"]
            for c in [0.1, 1.0, 10.0]
            for gamma in ["scale", 0.1, 1.0]
        ]

    if model_name == "RandomForest":
        return [
            (
                {
                    "n_estimators": 300,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_leaf,
                    "max_features": max_features,
                    "class_weight": cw,
                },
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    max_features=max_features,
                    class_weight=cw,
                    n_jobs=2,
                    random_state=split_seed,
                ),
            )
            for cw in [None, "balanced"]
            for max_depth in [3, 5, 8, None]
            for min_leaf in [1, 5, 10]
            for max_features in ["sqrt", None]
        ]

    if model_name == "HistGradientBoosting":
        return [
            (
                {
                    "learning_rate": lr,
                    "max_iter": max_iter,
                    "max_leaf_nodes": leaf_nodes,
                    "l2_regularization": l2,
                    "class_weight": cw,
                },
                HistGradientBoostingClassifier(
                    learning_rate=lr,
                    max_iter=max_iter,
                    max_leaf_nodes=leaf_nodes,
                    l2_regularization=l2,
                    class_weight=cw,
                    random_state=split_seed,
                ),
            )
            for cw in [None, "balanced"]
            for lr in [0.03, 0.1]
            for max_iter in [100, 200]
            for leaf_nodes in [15, 31]
            for l2 in [0.0, 0.1, 1.0]
        ]

    raise KeyError(f"Unknown model: {model_name}")


def _fit_select_eval(model_name, X_tr, y_tr, X_va, y_va, X_te, y_te, split_seed, max_hp=None):
    best = None
    candidates = _model_grid(model_name, split_seed)
    if max_hp is not None:
        candidates = candidates[:max_hp]

    for hp, model in candidates:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_tr, y_tr)
        val_pred, val_score = _score_and_pred(model, X_va)
        val_metrics = _metrics(y_va, val_pred, val_score)
        val_ts = val_metrics["TS"]
        if best is None or val_ts > best["val_metrics"]["TS"]:
            best = {
                "hp": hp,
                "model": model,
                "val_metrics": val_metrics,
            }

    test_pred, test_score = _score_and_pred(best["model"], X_te)
    test_metrics = _metrics(y_te, test_pred, test_score)
    return {
        "best_hp": best["hp"],
        "val_metrics": best["val_metrics"],
        "test_metrics": test_metrics,
    }


def _scale_block(arr: np.ndarray, train_idx, col_idx: list[int] | None = None) -> np.ndarray:
    block = arr if col_idx is None else arr[:, col_idx]
    tr_block = block[train_idx]
    mu = np.nanmean(tr_block, axis=0)
    sd = np.nanstd(tr_block, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    sd = np.where((np.isnan(sd)) | (sd < 1e-6), 1.0, sd)
    block = (block - mu) / sd
    return np.nan_to_num(block, nan=0.0)


def _build_feature_matrix(feature_set: str, df, train_idx, w_struct_raw=None) -> np.ndarray:
    X_n = build_n_tensor(df, train_idx).numpy()
    if feature_set == "n":
        return X_n
    if feature_set != "selected_aug":
        raise ValueError(f"Unknown feature_set={feature_set!r}")

    aug_np = df[SELECTED_AUG_COLS].values.astype(np.float32)
    aug_np = _scale_block(aug_np, train_idx)

    if w_struct_raw is None:
        raise ValueError("selected_aug feature set requires structured weather features")
    structw_idx = [STRUCTW_COLS.index(col) for col in SELECTED_STRUCTW_COLS]
    w_np = w_struct_raw.numpy().astype(np.float32)
    w_np = _scale_block(w_np, train_idx, structw_idx)

    return np.concatenate([X_n, aug_np, w_np], axis=1)


def main(train_ratio=None, models=None, split_seeds=None, output_tag=None, max_hp=None, feature_set="n"):
    selected_splits = [
        triplet for triplet in SPLIT_RATIOS
        if train_ratio is None or abs(triplet[0] - train_ratio) < 1e-9
    ]
    selected_models = models or MODEL_ORDER
    selected_split_seeds = split_seeds or DEFAULT_SPLIT_SEEDS

    invalid = sorted(set(selected_models) - set(MODEL_ORDER))
    if invalid:
        raise ValueError(f"Unknown models: {invalid}. Choices: {MODEL_ORDER}")

    if feature_set not in {"n", "selected_aug"}:
        raise ValueError("--feature-set must be one of: n, selected_aug")

    tag = _slugify(output_tag or "_".join(selected_models))
    progress_path = Path(f"results/literature_baseline_campaign_progress_{tag}.json")
    partial_path = Path(f"results/literature_baseline_campaign_partial_{tag}.json")
    output_path = Path(f"results/literature_baseline_campaign_{tag}.json")

    print("=== Literature Classical Baselines ===\n")
    print(f"Models: {selected_models}")
    print(f"Split ratios: {[label for _, _, label in selected_splits]}")
    print(f"Split seeds: {selected_split_seeds}")
    print(f"Feature set: {feature_set}")
    if max_hp is not None:
        print(f"Smoke/debug mode: limiting each model grid to first {max_hp} HP candidates")
    print()

    logger = create_run_logger(
        experiment_name="literature_baseline_campaign",
        config={
            "models": selected_models,
            "split_ratios": [label for _, _, label in selected_splits],
            "split_seeds": selected_split_seeds,
            "feature_set": feature_set,
            "feature_builder": (
                "features.build.build_n_tensor"
                if feature_set == "n"
                else "build_n_tensor + selected structured weather + selected augmented physical"
            ),
            "max_hp": max_hp,
            "selected_structw_cols": SELECTED_STRUCTW_COLS if feature_set == "selected_aug" else [],
            "selected_aug_cols": SELECTED_AUG_COLS if feature_set == "selected_aug" else [],
        },
        description="Classical literature baselines on the canonical N feature tensor.",
    )
    logger.write_notes(
        "# Literature baseline campaign\n\n"
        "- Uses `build_n_tensor(...)` after the canonical complete-case filter.\n"
        "- Optional selected_aug mode appends the same selected SW+AUG features used by NN+selected SW+AUG.\n"
        "- Always excludes R*, graph features, dense weather embeddings, and USGS-probability input features.\n"
        "- Hyperparameters are selected by validation Threat Score for each split.\n"
    )

    events = load_data()
    with open("data/augmented_features.json") as f:
        aug_raw = json.load(f)

    df_full = build_dataframe(events, aug_raw)
    complete_mask = df_full[CORE_FEATURES].notna().all(axis=1)
    orig_idx = df_full.index[complete_mask].tolist()
    ev_cc = [events[i] for i in orig_idx]
    aug_cc = {str(new_i): aug_raw.get(str(old_i), {}) for new_i, old_i in enumerate(orig_idx)}
    df = build_dataframe(ev_cc, aug_cc)
    y_all = df["pfdf"].values.astype(int)
    paired = [(ev, aug_cc.get(str(i), {})) for i, ev in enumerate(ev_cc)]
    w_struct_raw = None
    if feature_set == "selected_aug":
        w_struct_raw, coverage = load_weather_structured_raw(ev_cc)
        logger.log_event(
            "weather_structured_loaded",
            "Loaded event-level structured weather features for selected_aug mode",
            **coverage,
            selected_structw_cols=SELECTED_STRUCTW_COLS,
        )

    logger.log_event(
        "dataset_ready",
        "Prepared canonical complete-case dataframe",
        n_complete=len(df),
        n_positive=int(df["pfdf"].sum()),
        n_feature_dim=int(
            _build_feature_matrix(
                feature_set,
                df,
                list(range(min(10, len(df)))),
                w_struct_raw,
            ).shape[1]
        ),
    )

    started_at = time.time()
    completed = []
    raw_results = {
        label: {model_name: [] for model_name in selected_models}
        for _, _, label in selected_splits
    }

    for train_ratio_value, val_ratio, label in selected_splits:
        print(f"\n{'=' * 72}")
        print(f"Split {label}")
        print(f"{'=' * 72}")
        for model_name in selected_models:
            print(f"\n[{model_name}]")
            for split_seed in selected_split_seeds:
                print(f"  split_seed={split_seed}  tuning + test eval ...", flush=True)
                tr, va, te = zone_stratified_split_3way(
                    paired,
                    train_ratio=train_ratio_value,
                    val_ratio=val_ratio,
                    seed=split_seed,
                )
                X = _build_feature_matrix(feature_set, df, tr, w_struct_raw)
                y = y_all
                res = _fit_select_eval(
                    model_name,
                    X[tr], y[tr],
                    X[va], y[va],
                    X[te], y[te],
                    split_seed,
                    max_hp=max_hp,
                )
                record = {
                    "model": model_name,
                    "split_label": label,
                    "train_ratio": train_ratio_value,
                    "val_ratio": val_ratio,
                    "split_seed": split_seed,
                    "split_sizes": {
                        "train": len(tr),
                        "val": len(va),
                        "test": len(te),
                    },
                    "feature_builder": (
                        "build_n_tensor"
                        if feature_set == "n"
                        else "build_n_tensor + selected SW + selected AUG"
                    ),
                    "feature_set": feature_set,
                    "selected_structw_cols": SELECTED_STRUCTW_COLS if feature_set == "selected_aug" else [],
                    "selected_aug_cols": SELECTED_AUG_COLS if feature_set == "selected_aug" else [],
                    "feature_dim": int(X.shape[1]),
                    **res,
                }
                raw_results[label][model_name].append(record)
                completed.append({
                    "model": model_name,
                    "split_label": label,
                    "split_seed": split_seed,
                    "test_TS": record["test_metrics"]["TS"],
                    "best_hp": record["best_hp"],
                })
                _save_json(partial_path, raw_results)
                _save_json(
                    progress_path,
                    _build_progress_snapshot(
                        started_at,
                        logger.run_id,
                        selected_models,
                        [x[2] for x in selected_splits],
                        selected_split_seeds,
                        completed,
                    ),
                )
                logger.log_event(
                    "task_complete",
                    "Finished baseline task",
                    model=model_name,
                    split_label=label,
                    split_seed=split_seed,
                    test_TS=record["test_metrics"]["TS"],
                    best_hp=record["best_hp"],
                )

    aggregated = {
        label: {
            model_name: _aggregate_model_ratio(records)
            for model_name, records in per_model.items()
        }
        for label, per_model in raw_results.items()
    }
    _save_json(output_path, aggregated)
    logger.log_artifact(output_path, "result_json", "Aggregated literature baseline results")
    logger.log_artifact(partial_path, "partial_json", "Per-task literature baseline results")
    logger.log_artifact(progress_path, "progress_json", "Progress snapshot")
    logger.finalize(
        "completed",
        {
            "result_file": str(output_path),
            "partial_file": str(partial_path),
            "progress_file": str(progress_path),
        },
    )
    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None, choices=MODEL_ORDER)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument(
        "--feature-set",
        choices=["n", "selected_aug"],
        default="n",
        help="n = exact NN baseline features; selected_aug = N + selected SW + selected AUG.",
    )
    parser.add_argument(
        "--max-hp",
        type=int,
        default=None,
        help="Debug/smoke option: evaluate only the first N HP candidates.",
    )
    args = parser.parse_args()
    main(
        train_ratio=args.train_ratio,
        models=args.models,
        split_seeds=args.split_seeds,
        output_tag=args.output_tag,
        max_hp=args.max_hp,
        feature_set=args.feature_set,
    )
