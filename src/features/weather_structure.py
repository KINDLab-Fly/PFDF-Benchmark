"""
features/weather_structure.py — Convert NWS forecast text into structured features.

This module intentionally avoids document IDs, office IDs, and raw text tokens.
The goal is to keep only meteorologically meaningful, tabular signals that can
be compared more fairly against other structured predictors.
"""

from __future__ import annotations

import math
import re
from datetime import datetime

import pandas as pd

from .weather import load_nws_cache, weather_group_key, weather_lookup_key


WEATHER_STRUCT_COLS = [
    "wx_has_afd",
    "wx_has_hwo",
    "wx_has_ffw",
    "wx_lead_hours_afd",
    "wx_lead_hours_hwo",
    "wx_lead_hours_ffw",
    "wx_rain_amount_max_in",
    "wx_rain_rate_max_inph",
    "wx_heavy_rain_terms",
    "wx_flash_flood_terms",
    "wx_debris_flow_terms",
    "wx_burn_scar_terms",
    "wx_thunder_terms",
    "wx_uncertainty_terms",
    "wx_urgency_terms",
    "wx_watch_warning_flag",
    "wx_advisory_flag",
]

_AMOUNT_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(?:inch|inches|\")", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:inch|inches|\")", re.I),
]
_RATE_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(?:inch|inches|\")\s*(?:per hour|/hour|hr-1|an hour|in/hr)", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:inch|inches|\")\s*(?:per hour|/hour|hr-1|an hour|in/hr)", re.I),
]


def _storm_time(event):
    storm = event.get("StormStart") or event.get("StormDate")
    return pd.to_datetime(storm) if storm is not None else None


def _extract_timestamp(src: str):
    if not src or "@" not in src:
        return None
    try:
        return datetime.fromisoformat(src.split("@", 1)[1])
    except Exception:
        return None


def _lead_hours(event, src: str) -> float:
    storm = _storm_time(event)
    issued = _extract_timestamp(src)
    if storm is None or issued is None:
        return float("nan")
    return max((storm.to_pydatetime() - issued).total_seconds() / 3600.0, 0.0)


def _max_regex_value(text: str, patterns) -> float:
    best = 0.0
    for pattern in patterns:
        for match in pattern.findall(text):
            nums = [float(x) for x in match if x not in {"", None}]
            if nums:
                best = max(best, max(nums))
    return best


def _count_terms(text: str, phrases) -> float:
    return float(sum(text.count(phrase) for phrase in phrases))


def extract_weather_structured_record(event, nws_cache=None) -> dict:
    nws = nws_cache if nws_cache is not None else load_nws_cache()
    entry = nws.get(weather_lookup_key(event), {})

    afd = (entry.get("afd") or "").lower()
    hwo = (entry.get("hwo") or "").lower()
    ffw = (entry.get("ffw") or "").lower()
    all_text = "\n".join([afd, hwo, ffw])

    rec = {
        "wx_has_afd": float(bool(afd.strip())),
        "wx_has_hwo": float(bool(hwo.strip())),
        "wx_has_ffw": float(bool(ffw.strip())),
        "wx_lead_hours_afd": _lead_hours(event, entry.get("afd_src") or ""),
        "wx_lead_hours_hwo": _lead_hours(event, entry.get("hwo_src") or ""),
        "wx_lead_hours_ffw": _lead_hours(event, entry.get("ffw_src") or ""),
        "wx_rain_amount_max_in": _max_regex_value(all_text, _AMOUNT_PATTERNS),
        "wx_rain_rate_max_inph": _max_regex_value(all_text, _RATE_PATTERNS),
        "wx_heavy_rain_terms": _count_terms(all_text, [
            "heavy rain", "locally heavy rain", "moderate to heavy rain",
            "intense rainfall", "very heavy rain",
        ]),
        "wx_flash_flood_terms": _count_terms(all_text, [
            "flash flood", "flash flooding", "flooding", "flood threat",
        ]),
        "wx_debris_flow_terms": _count_terms(all_text, [
            "debris flow", "debris flows", "mudslide", "landslide",
        ]),
        "wx_burn_scar_terms": _count_terms(all_text, [
            "burn scar", "recent burn", "scar area", "fire area",
        ]),
        "wx_thunder_terms": _count_terms(all_text, [
            "thunderstorm", "thunderstorms", "convective", "convection",
        ]),
        "wx_uncertainty_terms": _count_terms(all_text, [
            "uncertain", "uncertainty", "possible", "may", "could", "chance",
        ]),
        "wx_urgency_terms": _count_terms(all_text, [
            "immediate", "dangerous", "urgent", "significant", "life threatening",
        ]),
        "wx_watch_warning_flag": float(
            any(token in all_text for token in ["watch", "warning"])
        ),
        "wx_advisory_flag": float("advisory" in all_text),
    }
    return rec


def build_weather_structure_dataframe(events: list, nws_cache=None) -> pd.DataFrame:
    nws = nws_cache if nws_cache is not None else load_nws_cache()
    rows = []
    for event in events:
        row = extract_weather_structured_record(event, nws)
        row["weather_group"] = str(weather_group_key(event, nws))
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in WEATHER_STRUCT_COLS:
        if col not in df:
            df[col] = 0.0
    return df[WEATHER_STRUCT_COLS + ["weather_group"]]
