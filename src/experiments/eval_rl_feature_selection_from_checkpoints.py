"""
RL feature selection from saved predictive-model checkpoints.

Loads checkpointed baseline models, freezes them, and learns independent
per-feature hard remove-mask logits with the class-collapse objective.
"""

from __future__ import annotations

import argparse
import json
import time
import math
import os
import pickle
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import NotFittedError

os.chdir(Path(__file__).resolve().parents[2])
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ZONE_ORDER
from experiments.eval_literature_baselines import _build_feature_matrix
from experiments.reproduce_checkpointed_baselines import FEATURE_CONFIGS
from features.build import CORE_FEATURES, N_ALL_COLS, build_dataframe
from features.load import load_data
from features.split import zone_stratified_split_3way
from features.weather_structured import load_weather_structured_raw
from models.mlp import EmbeddingMLP
from training.hp_search import SPLIT_RATIOS
from training.metrics import set_seed


DEFAULT_CLASSICAL_ENSEMBLE = ["FullLR", "PenalizedLR", "CART", "SVM", "RandomForest", "HistGradientBoosting"]

FEATURE_MEANINGS = {
    "x1": "Proportion of watershed burned at high/moderate severity and slope > 23 degrees",
    "x2": "Average dNBR divided by 1000, representing burn-severity magnitude",
    "x3": "Soil erodibility KF factor",
    "peak_i15": "Peak 15-minute rainfall intensity during the storm",
    "storm_dur": "Total storm duration in hours",
    "storm_accum": "Total storm rainfall accumulation",
    "storm_avg_i": "Average storm rainfall intensity",
    "log_area": "Log contributing watershed area",
    "gauge_dist": "Distance from rain gauge to documented response location",
    "acc030": "Peak 30-minute rainfall accumulation",
    "acc060": "Peak 60-minute rainfall accumulation",
    "month_sin": "Sine encoding of storm month",
    "month_cos": "Cosine encoding of storm month",
    "zone_ord": "Geographic zone category",
    "slope_deg": "Mean terrain slope from augmented terrain data",
    "soilm_30d": "Mean soil moisture over preceding three calendar months",
    "storm_duration_signal": "Structured forecast signal for storm duration or persistence",
    "antecedent_wetness_signal": "Structured forecast signal for antecedent wetness",
    "burn_scar_sensitivity_signal": "Structured forecast signal for burn-scar sensitivity",
    "mentions_burn_scar_or_debris_flow": "Forecast explicitly mentions burn scar or debris flow",
    "overall_pfdf_weather_risk": "Overall forecast-derived PFDF weather-risk score",
    "precipitation_signal": "Structured forecast signal for precipitation threat",
    "localized_extreme_signal": "Structured forecast signal for localized extreme rainfall",
}


class WrappedModel:
    def __init__(self, name, model, kind):
        self.name = name
        self.model = model
        self.kind = kind


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


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


def _split_from_label(ratio_label: str):
    for train_ratio, val_ratio, label in SPLIT_RATIOS:
        if label == ratio_label:
            return train_ratio, val_ratio, label
    raise ValueError(f"Unknown ratio label {ratio_label}")


def _checkpoint_dir(config: str, ratio_label: str, split_seed: int) -> Path:
    return Path("results/checkpoints/baselines") / config / _slug(ratio_label) / f"split_seed_{split_seed}"


def _load_models(config: str, ratio_label: str, split_seed: int, nn_eval_seed: int | None, classical_models: list[str]):
    cfg = FEATURE_CONFIGS[config]
    ckpt_dir = _checkpoint_dir(config, ratio_label, split_seed)
    models = []
    for model_name in classical_models:
        path = ckpt_dir / f"{model_name}.pkl"
        with path.open("rb") as f:
            payload = pickle.load(f)
        models.append(WrappedModel(model_name, payload["model"], "sklearn"))

    nn_seed = split_seed if nn_eval_seed is None else nn_eval_seed
    nn_path = ckpt_dir / f"{cfg['checkpoint_model_label']}_eval_seed_{nn_seed}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(nn_path, map_location=device)
    meta = payload["metadata"]
    hp = meta["best_hp"]
    model = EmbeddingMLP(len(meta["feature_names"]), hp["hidden"], hp["dropout"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    models.append(WrappedModel(cfg["checkpoint_model_label"], model, "torch"))
    return models, payload["metadata"]["feature_names"], ckpt_dir


def _predict_proba(wrapped: WrappedModel, X: np.ndarray) -> np.ndarray:
    if wrapped.kind == "torch":
        device = next(wrapped.model.parameters()).device
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=device)
            return torch.sigmoid(wrapped.model(xt)).cpu().numpy()
    if hasattr(wrapped.model, "predict_proba"):
        return wrapped.model.predict_proba(X)[:, 1]
    return 1.0 / (1.0 + np.exp(-wrapped.model.decision_function(X)))


def _threat_score(y_true, pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")


def _std_gap(pos: np.ndarray, neg: np.ndarray, eps: float = 1e-6) -> float:
    return abs(float(np.mean(pos)) - float(np.mean(neg))) / math.sqrt(float(np.var(pos)) + float(np.var(neg)) + eps)


def _masked_batch(X_group: np.ndarray, remove_masks: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return (1.0 - remove_masks[:, None, :]) * X_group[None, :, :] + remove_masks[:, None, :] * baseline[None, None, :]


def _baseline_vector(X: np.ndarray, train_idx, feature_names: list[str]) -> np.ndarray:
    baseline = np.zeros(X.shape[1], dtype=np.float32)
    if "zone_ord" in feature_names:
        zone_col = feature_names.index("zone_ord")
        zones = X[train_idx, zone_col].astype(int)
        if len(zones):
            values, counts = np.unique(zones, return_counts=True)
            baseline[zone_col] = float(values[np.argmax(counts)])
    return baseline


def _ensemble_scores(models: list[WrappedModel], X: np.ndarray) -> np.ndarray:
    return np.mean([_predict_proba(model, X) for model in models], axis=0)


def _mask_features(X: np.ndarray, baseline: np.ndarray, feature_names: list[str], remove_features: list[str]) -> np.ndarray:
    X_mask = X.copy()
    for feature in remove_features:
        j = feature_names.index(feature)
        X_mask[:, j] = baseline[j]
    return X_mask


def _train_policy(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    models: list[WrappedModel],
    baseline: np.ndarray,
    steps: int,
    samples_per_step: int,
    lr: float,
    sparsity_lambda: float,
    entropy_bonus: float,
    seed: int,
    progress_label: str | None = None,
):
    set_seed(seed)
    n_features = X_pos.shape[1]
    logits = torch.zeros(n_features, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    full_gaps = {}
    for model in models:
        full_gaps[model.name] = _std_gap(_predict_proba(model, X_pos), _predict_proba(model, X_neg))
    full_gap_mean = float(np.mean(list(full_gaps.values())))
    moving_baseline = None
    history = []
    last = {}

    for step in range(steps):
        probs = torch.sigmoid(logits)
        dist = torch.distributions.Bernoulli(probs=probs)
        masks_t = dist.sample((samples_per_step,))
        masks = masks_t.detach().cpu().numpy().astype(np.float32)
        pos_flat = _masked_batch(X_pos, masks, baseline).reshape(samples_per_step * len(X_pos), n_features)
        neg_flat = _masked_batch(X_neg, masks, baseline).reshape(samples_per_step * len(X_neg), n_features)

        gaps_by_model = []
        for model in models:
            pos_pred = _predict_proba(model, pos_flat).reshape(samples_per_step, len(X_pos))
            neg_pred = _predict_proba(model, neg_flat).reshape(samples_per_step, len(X_neg))
            gaps_by_model.append(np.array([_std_gap(pos_pred[i], neg_pred[i]) for i in range(samples_per_step)]))
        gap_matrix = np.stack(gaps_by_model, axis=1)
        d_mask = gap_matrix.mean(axis=1)
        collapse = 1.0 - d_mask / (full_gap_mean + 1e-6)
        removed = masks.sum(axis=1)
        rewards = collapse - sparsity_lambda * removed

        reward_t = torch.tensor(rewards, dtype=torch.float32)
        moving_baseline = float(reward_t.mean()) if moving_baseline is None else 0.9 * moving_baseline + 0.1 * float(reward_t.mean())
        loss = -((reward_t - moving_baseline).detach() * dist.log_prob(masks_t).sum(dim=1)).mean()
        loss = loss - entropy_bonus * dist.entropy().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        last = {
            "mean_reward": float(np.mean(rewards)),
            "mean_d_mask": float(np.mean(d_mask)),
            "mean_collapse": float(np.mean(collapse)),
            "mean_removed": float(np.mean(removed)),
            "hard_masks_verified": bool(np.all(np.isin(masks, [0.0, 1.0]))),
            "mean_d_mask_by_model": {model.name: float(np.mean(gap_matrix[:, i])) for i, model in enumerate(models)},
            "mean_collapse_by_model": {
                model.name: float(np.mean(1.0 - gap_matrix[:, i] / (full_gaps[model.name] + 1e-6)))
                for i, model in enumerate(models)
            },
        }
        if progress_label and ((step + 1) % 100 == 0 or step == 0 or step + 1 == steps):
            print(
                f"{progress_label} step={step + 1}/{steps} "
                f"reward={last['mean_reward']:.4f} "
                f"collapse={last['mean_collapse']:.4f} "
                f"removed={last['mean_removed']:.2f}",
                flush=True,
            )
        if step in {0, steps - 1} or (step + 1) % max(steps // 5, 1) == 0:
            history.append({"step": step + 1, **last, "mean_remove_prob": float(torch.sigmoid(logits).detach().mean())})

    remove_probs = torch.sigmoid(logits).detach().cpu().numpy()
    return {
        "logits": logits.detach().cpu().numpy(),
        "remove_probs": remove_probs,
        "full_gap_by_model": full_gaps,
        "full_gap_mean": full_gap_mean,
        "history": history,
        **last,
    }


def _load_policy_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    diagnostics = payload.get("diagnostics", {})
    metadata = payload.get("metadata", {})
    return {
        "remove_probs": payload["remove_probs"].detach().cpu().numpy(),
        "model_names": metadata.get("model_names"),
        "full_gap_by_model": diagnostics.get("full_gap_by_model", {}),
        "full_gap_mean": diagnostics.get("full_gap_mean", float("nan")),
        "mean_d_mask_by_model": diagnostics.get("mean_d_mask_by_model", {}),
        "mean_d_mask": diagnostics.get("mean_d_mask", float("nan")),
        "mean_collapse_by_model": diagnostics.get("mean_collapse_by_model", {}),
        "mean_collapse": diagnostics.get("mean_collapse", float("nan")),
        "mean_removed": diagnostics.get("mean_removed", float("nan")),
        "hard_masks_verified": diagnostics.get("hard_masks_verified", True),
    }


def _append_policy_outputs(
    *,
    rows: list[dict],
    diag: list[dict],
    top5_ts_rows: list[dict],
    config: str,
    ratio_label: str,
    split_seed: int,
    group_type: str,
    group_name: str,
    idx: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    policy: dict,
    policy_path: Path,
    per_model_rows: list[dict],
    X: np.ndarray,
    y: np.ndarray,
    models: list[WrappedModel],
    baseline: np.ndarray,
    feature_names: list[str],
):
    sorted_idx = np.argsort(-policy["remove_probs"])
    top5_features = [feature_names[i] for i in sorted_idx[:5]]
    X_group = X[idx]
    y_group = y[idx]
    full_pred = (_ensemble_scores(models, X_group) >= 0.5).astype(int)
    top5_pred = (_ensemble_scores(models, _mask_features(X_group, baseline, feature_names, top5_features)) >= 0.5).astype(int)
    ts_full = _threat_score(y_group, full_pred)
    ts_top5 = _threat_score(y_group, top5_pred)
    top5_ts_rows.append({
        "config": config,
        "ratio_label": ratio_label,
        "split_seed": split_seed,
        "group_type": group_type,
        "group_name": group_name,
        "n": int(len(idx)),
        "n_positive": int(len(pos_idx)),
        "n_negative": int(len(neg_idx)),
        "top5_features": "|".join(top5_features),
        "ts_full": ts_full,
        "ts_remove_top5": ts_top5,
        "delta_ts": ts_top5 - ts_full,
    })
    for i, feature in enumerate(feature_names):
        rows.append({
            "config": config,
            "ratio_label": ratio_label,
            "split_seed": split_seed,
            "group_type": group_type,
            "group_name": group_name,
            "feature": feature,
            "feature_index": i,
            "physical_meaning": FEATURE_MEANINGS.get(feature, feature),
            "remove_probability": float(policy["remove_probs"][i]),
            "policy_checkpoint": str(policy_path),
            "n_positive": int(len(pos_idx)),
            "n_negative": int(len(neg_idx)),
            "full_gap_mean": policy["full_gap_mean"],
            "mean_d_mask": policy["mean_d_mask"],
            "mean_collapse": policy["mean_collapse"],
            "mean_removed": policy["mean_removed"],
        })
    diag.append({
        "group_type": group_type,
        "group_name": group_name,
        "n_positive": int(len(pos_idx)),
        "n_negative": int(len(neg_idx)),
        "status": "completed",
        "policy_checkpoint": str(policy_path),
        "full_gap_mean": policy["full_gap_mean"],
        "mean_d_mask": policy["mean_d_mask"],
        "mean_collapse": policy["mean_collapse"],
        "mean_removed": policy["mean_removed"],
        "hard_masks_verified": policy["hard_masks_verified"],
    })
    for model_name, full_gap in policy.get("full_gap_by_model", {}).items():
        mean_d_mask = policy.get("mean_d_mask_by_model", {}).get(model_name, float("nan"))
        mean_collapse = policy.get("mean_collapse_by_model", {}).get(model_name, float("nan"))
        per_model_rows.append({
            "config": config,
            "ratio_label": ratio_label,
            "split_seed": split_seed,
            "group_type": group_type,
            "group_name": group_name,
            "model": model_name,
            "n_positive": int(len(pos_idx)),
            "n_negative": int(len(neg_idx)),
            "full_gap": full_gap,
            "mean_d_mask": mean_d_mask,
            "mean_collapse": mean_collapse,
        })


def _plot_region(df: pd.DataFrame, out_path: Path, value_col="remove_probability"):
    if df.empty:
        return
    pivot = df.pivot_table(index="group_name", columns="feature", values=value_col, aggfunc="mean")
    pivot = pivot.reindex([z for z in ZONE_ORDER if z in pivot.index])
    feature_order = df.groupby("feature")[value_col].mean().sort_values(ascending=False).index.tolist()
    pivot = pivot[feature_order]
    fig, ax = plt.subplots(figsize=(max(11, 0.45 * len(feature_order)), max(4, 0.4 * len(pivot.index) + 1.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(feature_order)))
    ax.set_xticklabels(feature_order, rotation=60, ha="right")
    ax.set_title("RL Feature Selection Region Masks")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run(config: str, ratio_label: str, split_seed: int, nn_eval_seed: int | None, steps: int, samples_per_step: int, min_class_count: int, resume: bool, classical_models: list[str]):
    cfg = FEATURE_CONFIGS[config]
    df, paired, w_struct_raw = _load_complete_case()
    train_ratio, val_ratio, _ = _split_from_label(ratio_label)
    tr, va, _ = zone_stratified_split_3way(paired, train_ratio=train_ratio, val_ratio=val_ratio, seed=split_seed)
    X = _build_feature_matrix(cfg["feature_set"], df, tr, w_struct_raw)
    y = df["pfdf"].values.astype(int)
    zones = df["zone"].values
    models, feature_names, model_ckpt_dir = _load_models(config, ratio_label, split_seed, nn_eval_seed, classical_models)
    model_names = [model.name for model in models]
    baseline = _baseline_vector(X, tr, feature_names)

    groups = [("global", "global", np.asarray(tr))]
    train_zones = zones[tr]
    for zone in ZONE_ORDER:
        groups.append(("region", zone, np.asarray(tr)[train_zones == zone]))

    rows = []
    diag = []
    per_model_rows = []
    policy_root = Path("results/checkpoints/rl_feature_selection") / config / _slug(ratio_label) / f"split_seed_{split_seed}"
    started_at = time.time()
    progress_path = Path(f"results/rl_feature_selection_{config}_{_slug(ratio_label)}_seed{split_seed}_progress.json")
    top5_ts_rows = []
    for group_type, group_name, idx in groups:
        pos_idx = idx[y[idx] == 1]
        neg_idx = idx[y[idx] == 0]
        if len(pos_idx) < min_class_count or len(neg_idx) < min_class_count:
            diag.append({"group_type": group_type, "group_name": group_name, "n_positive": len(pos_idx), "n_negative": len(neg_idx), "status": "skipped"})
            continue
        policy_path = policy_root / f"{group_type}_{group_name}.pt"
        if resume and policy_path.exists():
            policy = _load_policy_checkpoint(policy_path)
            if policy.get("model_names") != model_names:
                print(
                    f"{config} {ratio_label} seed={split_seed} recompute {group_type}:{group_name} "
                    f"because checkpoint model ensemble differs",
                    flush=True,
                )
            else:
                print(f"{config} {ratio_label} seed={split_seed} reuse {group_type}:{group_name} from {policy_path}", flush=True)
                _append_policy_outputs(
                    rows=rows,
                    diag=diag,
                    top5_ts_rows=top5_ts_rows,
                    config=config,
                    ratio_label=ratio_label,
                    split_seed=split_seed,
                    group_type=group_type,
                    group_name=group_name,
                    idx=idx,
                    pos_idx=pos_idx,
                    neg_idx=neg_idx,
                    policy=policy,
                    policy_path=policy_path,
                    per_model_rows=per_model_rows,
                    X=X,
                    y=y,
                    models=models,
                    baseline=baseline,
                    feature_names=feature_names,
                )
                continue
        print(f"{config} {ratio_label} seed={split_seed} optimize {group_type}:{group_name} pos={len(pos_idx)} neg={len(neg_idx)}", flush=True)
        seed = int.from_bytes(f"{config}|{ratio_label}|{split_seed}|{group_type}|{group_name}".encode()[:8].ljust(8, b"0"), "little") % (2**31)
        policy = _train_policy(
            X[pos_idx],
            X[neg_idx],
            models,
            baseline,
            steps=steps,
            samples_per_step=samples_per_step,
            lr=1e-2,
            sparsity_lambda=0.01,
            entropy_bonus=0.001,
            seed=seed,
            progress_label=f"{config} {ratio_label} seed={split_seed} {group_type}:{group_name}",
        )
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "logits": torch.tensor(policy["logits"], dtype=torch.float32),
            "remove_probs": torch.tensor(policy["remove_probs"], dtype=torch.float32),
            "feature_names": feature_names,
            "metadata": {
                "config": config,
                "ratio_label": ratio_label,
                "split_seed": split_seed,
                "nn_eval_seed": split_seed if nn_eval_seed is None else nn_eval_seed,
                "group_type": group_type,
                "group_name": group_name,
                "n_positive": int(len(pos_idx)),
                "n_negative": int(len(neg_idx)),
                "model_checkpoint_dir": str(model_ckpt_dir),
                "model_names": model_names,
                "objective": "class-collapse standardized mean gap",
            },
            "diagnostics": {k: v for k, v in policy.items() if k not in {"logits", "remove_probs"}},
        }, policy_path)
        _append_policy_outputs(
            rows=rows,
            diag=diag,
            top5_ts_rows=top5_ts_rows,
            config=config,
            ratio_label=ratio_label,
            split_seed=split_seed,
            group_type=group_type,
            group_name=group_name,
            idx=idx,
            pos_idx=pos_idx,
            neg_idx=neg_idx,
            policy=policy,
            policy_path=policy_path,
            per_model_rows=per_model_rows,
            X=X,
            y=y,
            models=models,
            baseline=baseline,
            feature_names=feature_names,
        )
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps({
            "config": config,
            "ratio_label": ratio_label,
            "split_seed": split_seed,
            "completed_groups": len([d for d in diag if d["status"] == "completed"]),
            "latest_group": group_name,
            "elapsed_sec": time.time() - started_at,
        }, indent=2))

    out_prefix = f"rl_feature_selection_{config}_{_slug(ratio_label)}_seed{split_seed}"
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(f"results/{out_prefix}_features.csv", index=False)
    pd.DataFrame(diag).to_csv(f"results/{out_prefix}_diagnostics.csv", index=False)
    pd.DataFrame(per_model_rows).to_csv(f"results/{out_prefix}_per_model_rewards.csv", index=False)
    pd.DataFrame(top5_ts_rows).to_csv(f"results/{out_prefix}_top5_ts.csv", index=False)
    _plot_region(rows_df[rows_df["group_type"] == "region"], Path(f"results/figures/{out_prefix}_region_heatmap.png"))
    print(f"Saved results/{out_prefix}_features.csv")


def aggregate(config: str, ratio_label: str, split_seeds: list[int]):
    slug_ratio = _slug(ratio_label)
    feature_parts = []
    top5_parts = []
    per_model_parts = []
    for seed in split_seeds:
        feat_path = Path(f"results/rl_feature_selection_{config}_{slug_ratio}_seed{seed}_features.csv")
        top5_path = Path(f"results/rl_feature_selection_{config}_{slug_ratio}_seed{seed}_top5_ts.csv")
        per_model_path = Path(f"results/rl_feature_selection_{config}_{slug_ratio}_seed{seed}_per_model_rewards.csv")
        if feat_path.exists() and feat_path.stat().st_size > 0:
            feature_parts.append(pd.read_csv(feat_path))
        if top5_path.exists() and top5_path.stat().st_size > 0:
            top5_parts.append(pd.read_csv(top5_path))
        if per_model_path.exists() and per_model_path.stat().st_size > 0:
            per_model_parts.append(pd.read_csv(per_model_path))
    if not feature_parts:
        raise SystemExit("No per-seed RL feature-selection outputs found.")
    feat = pd.concat(feature_parts, ignore_index=True)
    base_cols = ["group_type", "group_name", "feature", "feature_index", "physical_meaning"]
    feat_mean = (
        feat.groupby(base_cols, dropna=False)
        .agg(
            mean_remove_probability=("remove_probability", "mean"),
            std_remove_probability=("remove_probability", "std"),
            n_seed_records=("remove_probability", "count"),
            mean_full_gap=("full_gap_mean", "mean"),
            mean_collapse=("mean_collapse", "mean"),
            mean_removed=("mean_removed", "mean"),
            mean_n_positive=("n_positive", "mean"),
            mean_n_negative=("n_negative", "mean"),
        )
        .reset_index()
        .sort_values(["group_type", "group_name", "mean_remove_probability"], ascending=[True, True, False])
    )
    feat_mean.to_csv(f"results/rl_feature_selection_{config}_{slug_ratio}_feature_mean.csv", index=False)
    _plot_region(
        feat_mean[feat_mean["group_type"] == "region"],
        Path(f"results/figures/rl_feature_selection_{config}_{slug_ratio}_region_mean_heatmap.png"),
        value_col="mean_remove_probability",
    )
    if top5_parts:
        top5 = pd.concat(top5_parts, ignore_index=True)
        top5_summary = (
            top5.groupby(["group_type", "group_name"], dropna=False)
            .agg(
                n_mean=("n", "mean"),
                n_positive_mean=("n_positive", "mean"),
                n_negative_mean=("n_negative", "mean"),
                ts_full_mean=("ts_full", "mean"),
                ts_full_std=("ts_full", "std"),
                ts_remove_top5_mean=("ts_remove_top5", "mean"),
                ts_remove_top5_std=("ts_remove_top5", "std"),
                delta_ts_mean=("delta_ts", "mean"),
                delta_ts_std=("delta_ts", "std"),
                n_seed_records=("split_seed", "count"),
            )
            .reset_index()
            .sort_values(["group_type", "group_name"])
        )
        top5.to_csv(f"results/rl_feature_selection_{config}_{slug_ratio}_top5_ts_by_seed.csv", index=False)
        top5_summary.to_csv(f"results/rl_feature_selection_{config}_{slug_ratio}_top5_ts_summary.csv", index=False)
    if per_model_parts:
        per_model = pd.concat(per_model_parts, ignore_index=True)
        per_model_summary = (
            per_model.groupby(["group_type", "group_name", "model"], dropna=False)
            .agg(
                full_gap_mean=("full_gap", "mean"),
                mean_d_mask=("mean_d_mask", "mean"),
                mean_collapse=("mean_collapse", "mean"),
                n_seed_records=("split_seed", "count"),
            )
            .reset_index()
            .sort_values(["group_type", "group_name", "model"])
        )
        per_model.to_csv(f"results/rl_feature_selection_{config}_{slug_ratio}_per_model_rewards_by_seed.csv", index=False)
        per_model_summary.to_csv(f"results/rl_feature_selection_{config}_{slug_ratio}_per_model_rewards_summary.csv", index=False)
    print(f"Saved results/rl_feature_selection_{config}_{slug_ratio}_feature_mean.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "aggregate"], default="run")
    parser.add_argument("--config", choices=FEATURE_CONFIGS.keys(), default="augmented")
    parser.add_argument("--ratio", default="70/15/15")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--nn-eval-seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--samples-per-step", type=int, default=32)
    parser.add_argument("--min-class-count", type=int, default=1)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--resume", action="store_true", help="Reuse existing per-group policy checkpoints and continue missing groups.")
    parser.add_argument("--classical-models", nargs="+", default=DEFAULT_CLASSICAL_ENSEMBLE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "aggregate":
        aggregate(args.config, args.ratio, args.split_seeds)
    else:
        run(args.config, args.ratio, args.split_seed, args.nn_eval_seed, args.steps, args.samples_per_step, args.min_class_count, args.resume, args.classical_models)
