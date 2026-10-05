"""
experiments/eval_n_targeted_combo_campaign.py — Canonical evaluation for a
targeted N extension that combines selected structured-weather and augmented
physical features.

Model
-----
  - N+TargetedCombo : N plus
      * structured weather:
          precipitation_signal
          storm_duration_signal
          antecedent_wetness_signal
          burn_scar_sensitivity_signal
          localized_extreme_signal
          mentions_burn_scar_or_debris_flow
          overall_pfdf_weather_risk
      * augmented physical:
          soilm_30d
          slope_deg

Protocol mirrors the refreshed benchmark:
  - split ratios: 50/25/25, 60/20/20, 70/15/15, 80/10/10
  - outer split seeds: configurable, default 42-46
  - inner HP search / eval: training.hp_search.run_one
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

os.chdir(Path(__file__).resolve().parents[2])
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.logging_utils import create_run_logger
from features.build import CORE_FEATURES, build_dataframe, build_n_tensor
from features.load import load_data
from features.weather_structured import STRUCTW_COLS, load_weather_structured_raw
from models.mlp import EmbeddingMLP
from training.hp_search import SPLIT_RATIOS, run_one


MODEL_ORDER = ["N+TargetedCombo"]

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


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


def _aggregate_model_ratio(results_for_combo: list[dict]) -> dict:
    mean_scores = [r["mean_test"] for r in results_for_combo]
    return {
        "n_split_seeds": len(results_for_combo),
        "split_seed_results": results_for_combo,
        "mean_over_split_seeds": float(np.mean(mean_scores)),
        "std_over_split_seeds": float(np.std(mean_scores)),
    }


def _build_progress_snapshot(
    *,
    started_at: float,
    run_id: str,
    models: list[str],
    split_ratios: list[str],
    split_seeds: list[int],
    completed: list[dict],
) -> dict:
    total = len(models) * len(split_ratios) * len(split_seeds)
    done = len(completed)
    elapsed = time.time() - started_at
    rate = elapsed / done if done else None
    eta = rate * (total - done) if rate is not None else None
    latest = completed[-1] if completed else None
    return {
        "run_id": run_id,
        "timestamp_utc": _utc_now(),
        "total_tasks": total,
        "completed_tasks": done,
        "remaining_tasks": total - done,
        "elapsed_sec": elapsed,
        "eta_sec": eta,
        "latest_result": latest,
        "models": models,
        "split_ratios": split_ratios,
        "split_seeds": split_seeds,
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


def main(train_ratio=None, models=None, split_seeds=None, output_tag=None):
    selected_splits = [
        triplet for triplet in SPLIT_RATIOS
        if train_ratio is None or abs(triplet[0] - train_ratio) < 1e-9
    ]
    selected_models = models or MODEL_ORDER
    selected_split_seeds = split_seeds or [42, 43, 44, 45, 46]

    print("=== N+TargetedCombo — Targeted Structured Weather + Augmentation Campaign ===\n")
    print(f"Models: {selected_models}")
    print(f"Split ratios: {[label for _, _, label in selected_splits]}")
    print(f"Split seeds: {selected_split_seeds}\n")

    logger = create_run_logger(
        experiment_name="n_targeted_combo_campaign",
        config={
            "models": selected_models,
            "split_ratios": [label for _, _, label in selected_splits],
            "split_seeds": selected_split_seeds,
            "trainer": "training.hp_search.run_one",
            "selected_structw_cols": SELECTED_STRUCTW_COLS,
            "selected_aug_cols": SELECTED_AUG_COLS,
        },
        description=(
            "Canonical benchmark for N plus targeted structured-weather and "
            "augmented physical features."
        ),
    )
    logger.write_notes(
        "# Targeted structured weather + augmentation campaign\n\n"
        "- Goal: evaluate `N+TargetedCombo` under the same split-seed and HP-search protocol as the core benchmark.\n"
        "- Inputs = canonical `N` features + selected structured-weather features + selected augmented physical features.\n"
        "- Selected structured-weather features were chosen from the prior N+1 weather ablation; selected augmented physical features were chosen from the prior N+1 augmentation study.\n"
    )

    started_at = time.time()
    tag = output_tag or "_".join(selected_models)
    tag = _slugify(tag)
    progress_path = Path(f"results/n_targeted_combo_campaign_progress_{tag}.json")
    partial_results_path = Path(f"results/n_targeted_combo_campaign_partial_{tag}.json")

    events = load_data()
    logger.log_event("load_data", "Loaded raw events", n_events=len(events))
    with open("data/augmented_features.json") as f:
        aug_raw = json.load(f)

    df_full = build_dataframe(events, aug_raw)
    complete_mask = df_full[CORE_FEATURES].notna().all(axis=1)
    orig_idx = df_full.index[complete_mask].tolist()
    ev_cc = [events[i] for i in orig_idx]
    aug_cc = {str(new_i): aug_raw.get(str(old_i), {}) for new_i, old_i in enumerate(orig_idx)}
    df = build_dataframe(ev_cc, aug_cc)
    y_all = torch.tensor(df["pfdf"].values, dtype=torch.float32)
    paired = [(ev, aug_cc.get(str(i), {})) for i, ev in enumerate(ev_cc)]
    logger.log_event(
        "dataset_ready",
        "Prepared complete-case dataframe for targeted combo study",
        n_complete=len(df),
        n_positive=int(df["pfdf"].sum()),
    )

    print("Loading structured weather features for N+TargetedCombo …")
    W_struct_raw, coverage = load_weather_structured_raw(ev_cc)
    structw_idx = [STRUCTW_COLS.index(col) for col in SELECTED_STRUCTW_COLS]
    logger.log_event(
        "weather_structured_loaded",
        "Loaded event-level structured weather features",
        **coverage,
    )
    print(
        f"  coverage: {coverage['n_covered']}/{coverage['n_events']} "
        f"({100*coverage['coverage_frac']:.1f}%), n_selected_features={len(structw_idx)}"
    )

    def _get_splits(tr, va, te):
        X_n = build_n_tensor(df, tr)

        aug_np = df[SELECTED_AUG_COLS].values.astype(np.float32)
        aug_np = _scale_block(aug_np, tr)
        X_aug = torch.tensor(aug_np, dtype=torch.float32)

        W_np = W_struct_raw.numpy().astype(np.float32)
        W_np = _scale_block(W_np, tr, structw_idx)
        X_structw = torch.tensor(W_np, dtype=torch.float32)

        X = torch.cat([X_n, X_aug, X_structw], dim=1)
        return X[tr], y_all[tr], X[va], y_all[va], X[te], y_all[te]

    def factory(feat_dim):
        return lambda hp: EmbeddingMLP(feat_dim, hp["hidden"], hp["dropout"])

    model_registry = {
        "N+TargetedCombo": _get_splits,
    }

    completed = []
    raw_results = {
        label: {model_name: [] for model_name in selected_models}
        for _, _, label in selected_splits
    }

    for train_ratio_value, val_ratio, label in selected_splits:
        print(f"\n{'='*72}")
        print(f"Split {label}")
        print(f"{'='*72}")
        for model_name in selected_models:
            get_splits = model_registry[model_name]
            print(f"\n[{model_name}]")
            for split_seed in selected_split_seeds:
                print(f"  split_seed={split_seed}  HP search + final eval …", flush=True)
                logger.log_event(
                    "task_start",
                    "Starting targeted-combo model/ratio/split_seed task",
                    model=model_name,
                    split_label=label,
                    split_seed=split_seed,
                    train_ratio=train_ratio_value,
                    val_ratio=val_ratio,
                )
                res = run_one(
                    get_splits,
                    factory,
                    paired,
                    train_ratio_value,
                    val_ratio,
                    split_seed=split_seed,
                )
                task_record = {
                    "model": model_name,
                    "split_label": label,
                    "train_ratio": train_ratio_value,
                    "val_ratio": val_ratio,
                    "split_seed": split_seed,
                    "selected_structw_cols": SELECTED_STRUCTW_COLS,
                    "selected_aug_cols": SELECTED_AUG_COLS,
                    **res,
                }
                raw_results[label][model_name].append(task_record)
                completed.append({
                    "model": model_name,
                    "split_label": label,
                    "split_seed": split_seed,
                    "mean_test": res["mean_test"],
                    "std_test": res["std_test"],
                })
                logger.log_event(
                    "task_complete",
                    "Finished targeted-combo model/ratio/split_seed task",
                    model=model_name,
                    split_label=label,
                    split_seed=split_seed,
                    mean_test=res["mean_test"],
                    std_test=res["std_test"],
                    best_hp=res["best_hp"],
                    split_sizes=res["split_sizes"],
                )
                print(
                    f"    mean={res['mean_test']:.4f}  std={res['std_test']:.4f}  "
                    f"best_hp={res['best_hp']}"
                )
                progress = _build_progress_snapshot(
                    started_at=started_at,
                    run_id=logger.run_id,
                    models=selected_models,
                    split_ratios=[split_label for _, _, split_label in selected_splits],
                    split_seeds=selected_split_seeds,
                    completed=completed,
                )
                _save_json(progress_path, progress)
                _save_json(partial_results_path, raw_results)

    final_summary = {}
    print(f"\n{'='*72}")
    print("Final aggregated results")
    print(f"{'='*72}")
    for _, _, label in selected_splits:
        print(f"\n{label}")
        final_summary[label] = {}
        for model_name in selected_models:
            agg = _aggregate_model_ratio(raw_results[label][model_name])
            final_summary[label][model_name] = agg
            print(
                f"  {model_name:<20}  "
                f"mean={agg['mean_over_split_seeds']:.4f}  "
                f"std={agg['std_over_split_seeds']:.4f}"
            )

    output_filename = (
        f"n_targeted_combo_eval_{tag}.json"
        if output_tag else "n_targeted_combo_eval.json"
    )
    output_path = Path("results") / output_filename
    _save_json(output_path, final_summary)
    logger.log_artifact(output_path, "result_json", "Aggregated targeted-combo results")
    logger.log_artifact(progress_path, "progress_json", "Live task progress snapshot")
    logger.log_artifact(partial_results_path, "partial_result_json", "Live per-task result table")
    logger.finalize(
        "completed",
        {
            "result_file": str(output_path),
            "progress_file": str(progress_path),
            "partial_results_file": str(partial_results_path),
            "models": selected_models,
            "split_ratios": [label for _, _, label in selected_splits],
            "split_seeds": selected_split_seeds,
            "selected_structw_cols": SELECTED_STRUCTW_COLS,
            "selected_aug_cols": SELECTED_AUG_COLS,
        },
    )
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Optional single train ratio, e.g. 0.70",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=MODEL_ORDER,
        help="Optional subset of models to run",
    )
    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional outer split seeds",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Optional tag used in live progress filenames",
    )
    args = parser.parse_args()
    main(
        train_ratio=args.train_ratio,
        models=args.models,
        split_seeds=args.split_seeds,
        output_tag=args.output_tag,
    )
