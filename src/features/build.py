"""
features/build.py — Feature building for all models.

Provides:
  - FEATURE_DEFS / ALL_FEATURES / CORE_FEATURES  : 20-node graph feature definitions
  - build_dataframe(events, aug, rain)            : events → pandas DataFrame
  - prepare_tensors(df, feature_cols, tr, va, te) : z-score on train → (X,y) tensors

  - N_CONT_COLS / N_EXTRA_COLS / AUG_COLS_GS      : column name constants
  - build_n_tensor(df, train_idx)                 : 14-dim N feature tensor
  - build_naug_tensor(df, train_idx)              : 19-dim N_aug feature tensor
  - build_nrstar_tensor(df, train_idx)            : N + rainfall anomaly tensor
  - build_naug_rstar_tensor(df, train_idx)        : N_aug + rainfall anomaly tensor
"""

import math
import numpy as np
import pandas as pd
import torch
from config import ZONE_ORDER

# ─────────────────────────────────────────────────────────────────────────────
# Feature node definitions — 20-node graph (19 input nodes + pfdf target)
# ─────────────────────────────────────────────────────────────────────────────

def _sf(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")

_ZONE_IDX = {z: i for i, z in enumerate(ZONE_ORDER)}


def _event_month(event) -> int:
    for key in ["StormStart", "StormDate"]:
        value = event.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in {"nan", "nat", ""}:
            continue
        try:
            return int(pd.Timestamp(value).month)
        except Exception:
            continue
    return 6

FEATURE_DEFS = {
    # Original data features
    "storm_accum":     lambda e, _: _sf(e["StormAccum_mm"]),
    "storm_dur":       lambda e, _: _sf(e["StormDur_H"]),
    "storm_avg_i":     lambda e, _: _sf(e["StormAvgI_mmph"]),
    "peak_i15":        lambda e, _: _sf(e["Peak_I15_mmph"]),
    "acc030":          lambda e, _: _sf(e["Acc030_mm"]),
    "acc060":          lambda e, _: _sf(e["Acc060_mm"]),
    "log_area":        lambda e, _: math.log(e["ContribArea_km2"])
                                    if e.get("ContribArea_km2") and e["ContribArea_km2"] > 0
                                    else float("nan"),
    "x1":              lambda e, _: _sf(e["X1"]),
    "x2":              lambda e, _: _sf(e["X2"]),
    "x3":              lambda e, _: _sf(e["X3"]),
    "month_sin":       lambda e, _: math.sin(2 * math.pi * _event_month(e) / 12),
    "month_cos":       lambda e, _: math.cos(2 * math.pi * _event_month(e) / 12),
    "zone_ord":        lambda e, _: float(_ZONE_IDX.get(e.get("Zone", ""), 0)),
    "gauge_dist":      lambda e, _: _sf(e["GaugeDist_m"]),
    # Augmented features (from augmented_features.json)
    "soilm_7d":        lambda e, aug: _sf(aug.get("soilm_7d_mean_kg_m2",  float("nan"))),
    "soilm_30d":       lambda e, aug: _sf(aug.get("soilm_30d_mean_kg_m2", float("nan"))),
    "slope_deg":       lambda e, aug: _sf(aug.get("slope_mean_deg",        float("nan"))),
    "aspect_deg":      lambda e, aug: _sf(aug.get("aspect_mean_deg",       float("nan"))),
    "time_since_fire": lambda e, aug: _sf(aug.get("time_since_fire_days",  float("nan"))),
}

# CORE_FEATURES: original-data features used for complete-case filtering
CORE_FEATURES = [
    "storm_accum", "storm_dur", "peak_i15", "acc030", "acc060",
    "log_area", "x1", "x2", "x3", "month_sin", "month_cos",
]
ALL_FEATURES = list(FEATURE_DEFS.keys())   # 19 input feature nodes


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame builder
# ─────────────────────────────────────────────────────────────────────────────

def _rainfall_anomaly_row(event: dict, rain: dict) -> dict:
    observed_i15 = _sf(event.get("Peak_I15_mmph", float("nan")))
    local_i15 = _sf(rain.get("i15_1yr_mmph", float("nan")))
    rstar = _sf(rain.get("rstar_i15", float("nan")))

    if not math.isfinite(rstar) and math.isfinite(observed_i15) and local_i15 > 0:
        rstar = observed_i15 / local_i15

    log_rstar = math.log(rstar) if math.isfinite(rstar) and rstar > 0 else float("nan")

    return {
        "i15_1yr": local_i15,
        "rstar_i15": rstar,
        "log_rstar_i15": log_rstar,
    }


def build_dataframe(events: list, aug_features: dict, rain_features: dict | None = None) -> pd.DataFrame:
    """
    Convert event list + augmented feature/rainfall climatology caches → DataFrame.

    Columns: pfdf (label), zone (str), usgs_p, one column per FEATURE_DEFS entry,
    plus optional rainfall anomaly columns.

    aug_features and rain_features are keyed by str(original_event_index).
    """
    rain_features = rain_features or {}
    rows = []
    for idx, e in enumerate(events):
        aug = aug_features.get(str(idx), {})
        rain = rain_features.get(str(idx), {})
        row = {
            "pfdf":   float(e["Response"]),
            "zone":   e.get("Zone", ""),
            "usgs_p": _sf(e.get("P", float("nan"))),
            "acc015": _sf(e.get("Acc015_mm", e.get("R", float("nan")))),
        }
        for name, fn in FEATURE_DEFS.items():
            try:
                row[name] = fn(e, aug)
            except Exception:
                row[name] = float("nan")
        row.update(_rainfall_anomaly_row(e, rain))
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_tensors(df, feature_cols, train_idx, val_idx, test_idx):
    """
    Z-score feature_cols on train split, fill NaN → 0.

    Returns ((X_tr, y_tr), (X_va, y_va), (X_te, y_te)) float32 tensors.
    """
    means = df.iloc[train_idx][feature_cols].mean()
    stds  = df.iloc[train_idx][feature_cols].std().replace(0, 1)

    def _split(idx):
        X = (df.iloc[idx][feature_cols] - means) / stds
        X = X.fillna(0.0)
        y = df["pfdf"].values[idx]
        return (
            torch.tensor(X.values, dtype=torch.float32),
            torch.tensor(y,        dtype=torch.float32),
        )

    return _split(train_idx), _split(val_idx), _split(test_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Feature column constants for N and N_aug
# ─────────────────────────────────────────────────────────────────────────────

# N (14-dim): 11 continuous z-scored + month_sin + month_cos + zone_ord
N_CONT_COLS  = ["x1", "x2", "x3", "peak_i15", "storm_dur", "storm_accum",
                "storm_avg_i", "log_area", "gauge_dist", "acc030", "acc060"]
N_EXTRA_COLS = ["month_sin", "month_cos", "zone_ord"]
N_ALL_COLS   = N_CONT_COLS + N_EXTRA_COLS           # 14 total; zone_ord at index 13

# N_aug (19-dim): N + 5 augmented features; zone_ord still at index 13
AUG_COLS_GS  = ["soilm_7d", "soilm_30d", "slope_deg", "aspect_deg", "time_since_fire"]
NAUG_ALL_COLS = N_ALL_COLS + AUG_COLS_GS            # 19 total

# Rainfall anomaly features from the USGS 1-year I15 climatology raster.
RSTAR_COLS = ["i15_1yr", "rstar_i15", "log_rstar_i15"]
NRSTAR_ALL_COLS = N_ALL_COLS + RSTAR_COLS
NAUG_RSTAR_ALL_COLS = NAUG_ALL_COLS + RSTAR_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Tensor builders
# ─────────────────────────────────────────────────────────────────────────────

def build_n_tensor(df: pd.DataFrame, train_idx) -> torch.Tensor:
    """
    Build 14-dim N feature tensor.
    Z-scores 11 continuous features on train; leaves month cyclic and zone_ord as-is.
    zone_ord (index 13) is left as float 0–9 for embedding lookup by the model.
    """
    X = df[N_ALL_COLS].values.astype(np.float32)
    cont_idx = list(range(len(N_CONT_COLS)))
    mu = X[train_idx][:, cont_idx].mean(axis=0)
    sd = X[train_idx][:, cont_idx].std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    X[:, cont_idx] = (X[:, cont_idx] - mu) / sd
    return torch.tensor(X, dtype=torch.float32)


def build_naug_tensor(df: pd.DataFrame, train_idx) -> torch.Tensor:
    """
    Build 19-dim N_aug feature tensor.
    Z-scores continuous and augmented features on train; NaN aug → 0.
    zone_ord (index 13) is left as float 0–9 for embedding lookup by the model.
    """
    X = df[NAUG_ALL_COLS].values.astype(np.float32)
    cont_idx = list(range(len(N_CONT_COLS)))
    aug_idx  = list(range(len(N_ALL_COLS), len(N_ALL_COLS) + len(AUG_COLS_GS)))

    for idx in [cont_idx, aug_idx]:
        tr_block = X[train_idx][:, idx]
        mu = np.nanmean(tr_block, axis=0)
        sd = np.nanstd(tr_block,  axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        X[:, idx] = (X[:, idx] - mu) / sd

    X = np.nan_to_num(X, nan=0.0)
    return torch.tensor(X, dtype=torch.float32)


def build_naug_subset_tensor(df: pd.DataFrame, train_idx, aug_cols) -> torch.Tensor:
    """
    Build N-style structured features plus a selected subset of augmented columns.

    Parameters
    ----------
    aug_cols : list[str]
        Selected augmented columns from AUG_COLS_GS.

    Notes
    -----
    The first 14 columns match build_n_tensor exactly, so zone_ord remains at
    index 13 for EmbeddingMLP.
    """
    feature_cols = N_ALL_COLS + list(aug_cols)
    X = df[feature_cols].values.astype(np.float32)
    cont_idx = list(range(len(N_CONT_COLS)))
    aug_start = len(N_ALL_COLS)
    aug_idx = list(range(aug_start, aug_start + len(aug_cols)))

    for idx in [cont_idx, aug_idx]:
        if not idx:
            continue
        tr_block = X[train_idx][:, idx]
        mu = np.nanmean(tr_block, axis=0)
        sd = np.nanstd(tr_block, axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        X[:, idx] = (X[:, idx] - mu) / sd

    X = np.nan_to_num(X, nan=0.0)
    return torch.tensor(X, dtype=torch.float32)


def _zscore_column_blocks(X: np.ndarray, train_idx, blocks: list[list[int]]) -> np.ndarray:
    for idx in blocks:
        if not idx:
            continue
        tr_block = X[train_idx][:, idx]
        mu = np.nanmean(tr_block, axis=0)
        sd = np.nanstd(tr_block, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where((np.isfinite(sd)) & (sd >= 1e-6), sd, 1.0)
        X[:, idx] = (X[:, idx] - mu) / sd
    return np.nan_to_num(X, nan=0.0)


def build_nrstar_tensor(df: pd.DataFrame, train_idx) -> torch.Tensor:
    """
    Build N + rainfall anomaly tensor.

    Adds local 1-year 15-minute intensity, R*_15, and log(R*_15).  The zone
    ordinal remains at index 13 for EmbeddingMLP.
    """
    X = df[NRSTAR_ALL_COLS].values.astype(np.float32)
    cont_idx = list(range(len(N_CONT_COLS)))
    rstar_start = len(N_ALL_COLS)
    rstar_idx = list(range(rstar_start, rstar_start + len(RSTAR_COLS)))
    X = _zscore_column_blocks(X, train_idx, [cont_idx, rstar_idx])
    return torch.tensor(X, dtype=torch.float32)


def build_naug_rstar_tensor(df: pd.DataFrame, train_idx) -> torch.Tensor:
    """
    Build N_aug + rainfall anomaly tensor.

    The first 19 columns match build_naug_tensor exactly; the final 3 columns
    are the local I15 climatology and derived anomaly features.
    """
    X = df[NAUG_RSTAR_ALL_COLS].values.astype(np.float32)
    cont_idx = list(range(len(N_CONT_COLS)))
    aug_idx = list(range(len(N_ALL_COLS), len(N_ALL_COLS) + len(AUG_COLS_GS)))
    rstar_start = len(NAUG_ALL_COLS)
    rstar_idx = list(range(rstar_start, rstar_start + len(RSTAR_COLS)))
    X = _zscore_column_blocks(X, train_idx, [cont_idx, aug_idx, rstar_idx])
    return torch.tensor(X, dtype=torch.float32)
