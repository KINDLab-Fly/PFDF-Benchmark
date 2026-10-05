"""
features/weather_structured.py — Structured weather features extracted from forecast text.

Provides:
  - STRUCTW_COLS                       : numeric weather-derived feature names
  - load_weather_structured_raw(events): raw (possibly NaN) tensor aligned to events
  - build_structw_tensor(events, tr)   : z-scored on train split, NaN -> 0
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .weather import weather_lookup_key
from config import RELEASE_ROOT


WEATHER_STRUCTURED_EVENT_FILE = str(RELEASE_ROOT / "data/weather_structured_event_afd.json")

STRUCTW_COLS = [
    "precipitation_signal",
    "short_duration_intensity_signal",
    "storm_duration_signal",
    "training_repetition_signal",
    "antecedent_wetness_signal",
    "convective_signal",
    "orographic_enhancement_signal",
    "explicit_flash_flood_signal",
    "burn_scar_sensitivity_signal",
    "localized_extreme_signal",
    "mentions_heavy_rain",
    "mentions_thunderstorm",
    "mentions_flash_flood",
    "mentions_burn_scar_or_debris_flow",
    "mentions_training_or_repeating_cells",
    "mentions_slow_storm_motion",
    "mentions_monsoon_or_tropical_moisture",
    "overall_pfdf_weather_risk",
]


def load_weather_structured_event_cache(path: str = WEATHER_STRUCTURED_EVENT_FILE) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"'{path}' not found. Run python extract_weather_structured.py first."
        )
    with p.open() as f:
        return json.load(f)


def load_weather_structured_raw(events: list, path: str = WEATHER_STRUCTURED_EVENT_FILE):
    """
    Load numeric structured weather features aligned to `events`.

    Returns
    -------
    (X_raw, coverage)
      X_raw    : (N, len(STRUCTW_COLS)) float32 tensor, may contain NaNs for missing
      coverage : dict with basic coverage stats
    """
    cache = load_weather_structured_event_cache(path)

    rows = []
    covered = 0
    for event in events:
        key = weather_lookup_key(event)
        entry = cache.get(key, {})
        feats = entry.get("features", {})
        if feats:
            covered += 1
        rows.append([
            float(feats.get(col, float("nan"))) if feats.get(col) is not None else float("nan")
            for col in STRUCTW_COLS
        ])

    arr = np.asarray(rows, dtype=np.float32)
    coverage = {
        "n_events": len(events),
        "n_covered": covered,
        "coverage_frac": (covered / len(events)) if events else 0.0,
        "n_features": len(STRUCTW_COLS),
    }
    return torch.tensor(arr, dtype=torch.float32), coverage


def build_structw_tensor(
    events: list,
    train_idx,
    path: str = WEATHER_STRUCTURED_EVENT_FILE,
) -> torch.Tensor:
    """
    Build z-scored structured-weather tensor aligned to `events`.

    All structured weather columns are standardized on the train split and
    missing values are filled with 0 after scaling.
    """
    X_raw, _ = load_weather_structured_raw(events, path=path)
    X = X_raw.numpy().astype(np.float32)

    tr_block = X[train_idx]
    mu = np.nanmean(tr_block, axis=0)
    sd = np.nanstd(tr_block, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    sd = np.where((np.isnan(sd)) | (sd < 1e-6), 1.0, sd)

    X = (X - mu) / sd
    X = np.nan_to_num(X, nan=0.0)
    return torch.tensor(X, dtype=torch.float32)
