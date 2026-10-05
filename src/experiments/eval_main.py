"""
experiments/eval_main.py — Canonical N / N_aug / N-G comparison campaign.

This is the refreshed benchmark runner for the core models:
  - N
  - N_aug
  - N-G   (renamed from G*-MLP)

Protocol
--------
- Ratios: 50/25/25, 60/20/20, 70/15/15, 80/10/10
- Split seeds: configurable, default 42-46
- For each (model, ratio, split_seed):
    * fixed split by split_seed
    * 200-config HP search
    * 5 HP-selection init seeds
    * 10 final-eval init seeds

Progress artifacts are written continuously so long-running campaigns can be
checked while running in tmux.
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
from features.build import (
    CORE_FEATURES,
    NAUG_ALL_COLS,
    build_dataframe,
    build_n_tensor,
    build_naug_tensor,
)
from features.load import load_data
from models.baselines import usgs_threat_score
from models.graph_mlp import EmbeddingGraphMLP
from models.mlp import EmbeddingMLP
from training.hp_search import SPLIT_RATIOS, run_one


MODEL_ORDER = ["N", "N_aug", "N-G"]


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


def main(train_ratio=None, models=None, split_seeds=None, output_tag=None):
    selected_splits = [
        triplet for triplet in SPLIT_RATIOS
        if train_ratio is None or abs(triplet[0] - train_ratio) < 1e-9
    ]
    selected_models = models or MODEL_ORDER
    selected_split_seeds = split_seeds or [42, 43, 44, 45, 46]

    print("=== N / N_aug / N-G — Canonical Evaluation Campaign ===\n")
    print(f"Models: {selected_models}")
    print(f"Split ratios: {[label for _, _, label in selected_splits]}")
    print(f"Split seeds: {selected_split_seeds}\n")

    logger = create_run_logger(
        experiment_name="core_benchmark_campaign",
        config={
            "models": selected_models,
            "split_ratios": [label for _, _, label in selected_splits],
            "split_seeds": selected_split_seeds,
            "trainer": "training.hp_search.run_one",
        },
        description="Core benchmark campaign across ratios and split seeds for N, N_aug, and N-G.",
    )
    logger.write_notes(
        "# Core benchmark campaign\n\n"
        "- Goal: run `N`, `N_aug`, and `N-G` across all requested split ratios and split seeds.\n"
        "- Progress is written continuously so the run can be monitored in tmux.\n"
        "- Final summary aggregates over split seeds for each ratio/model pair.\n"
    )

    started_at = time.time()
    tag = output_tag or "_".join(selected_models)
    tag = _slugify(tag)
    progress_path = Path(f"results/core_benchmark_campaign_progress_{tag}.json")
    partial_results_path = Path(f"results/core_benchmark_campaign_partial_{tag}.json")

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
        "complete_case_filter",
        "Built complete-case dataframe",
        n_complete=len(df),
        n_positive=int(df["pfdf"].sum()),
        usgs_complete_case_ts=usgs_threat_score(df),
    )

    def n_get_splits(tr, va, te):
        X = build_n_tensor(df, tr)
        return X[tr], y_all[tr], X[va], y_all[va], X[te], y_all[te]

    def naug_get_splits(tr, va, te):
        X = build_naug_tensor(df, tr)
        return X[tr], y_all[tr], X[va], y_all[va], X[te], y_all[te]

    def n_factory(feat_dim):
        return lambda hp: EmbeddingMLP(feat_dim, hp["hidden"], hp["dropout"])

    def naug_factory(feat_dim):
        return lambda hp: EmbeddingMLP(feat_dim, hp["hidden"], hp["dropout"])

    def ng_factory(feat_dim):
        return lambda hp: EmbeddingGraphMLP(
            feat_dim, NAUG_ALL_COLS, hp["hidden"], hp["dropout"]
        )

    model_registry = {
        "N": (n_get_splits, n_factory),
        "N_aug": (naug_get_splits, naug_factory),
        "N-G": (naug_get_splits, ng_factory),
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
            get_splits, factory = model_registry[model_name]
            print(f"\n[{model_name}]")
            for split_seed in selected_split_seeds:
                print(f"  split_seed={split_seed}  HP search + final eval …", flush=True)
                logger.log_event(
                    "task_start",
                    "Starting model/ratio/split_seed task",
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
                    "Finished model/ratio/split_seed task",
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
                    split_ratios=[sr for _, _, sr in selected_splits],
                    split_seeds=selected_split_seeds,
                    completed=completed,
                )
                _save_json(progress_path, progress)
                _save_json(partial_results_path, raw_results)

    final_results = {}
    for _, _, label in selected_splits:
        final_results[label] = {}
        for model_name in selected_models:
            final_results[label][model_name] = _aggregate_model_ratio(raw_results[label][model_name])

    out_path = Path(f"results/core_benchmark_campaign_{tag}.json")
    _save_json(out_path, final_results)
    logger.log_artifact(out_path, "result_json", "Core benchmark campaign final results")
    logger.log_artifact(progress_path, "progress_json", "Live progress snapshot")
    logger.log_artifact(partial_results_path, "partial_json", "Incremental raw results")

    print(f"\nSaved → {out_path}")
    print("\n=== Summary: mean ± std over split seeds ===")
    print(f"{'Split':<12}" + "".join(f"  {m:<22}" for m in selected_models))
    print("-" * (12 + 24 * len(selected_models)))
    for _, _, label in selected_splits:
        row = f"{label:<12}"
        for model_name in selected_models:
            r = final_results[label][model_name]
            row += f"  {r['mean_over_split_seeds']:.4f}±{r['std_over_split_seeds']:.4f}         "
        print(row)

    logger.finalize(
        "completed",
        {
            "result_file": str(out_path),
            "progress_file": str(progress_path),
            "partial_results_file": str(partial_results_path),
            "models": selected_models,
            "splits": list(final_results.keys()),
            "split_seeds": selected_split_seeds,
        },
    )


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
        choices=MODEL_ORDER,
        default=None,
        help="Optional subset of models",
    )
    parser.add_argument(
        "--split-seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Outer split seeds to evaluate",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Tag appended to progress/result filenames to avoid collisions.",
    )
    args = parser.parse_args()
    main(
        train_ratio=args.train_ratio,
        models=args.models,
        split_seeds=args.split_seeds,
        output_tag=args.output_tag,
    )
