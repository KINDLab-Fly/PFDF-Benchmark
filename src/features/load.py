"""
features/load.py — Load raw USGS PFDF dataset from Excel.

Provides load_data() which returns a list of event dicts (1,243 events).
Each dict contains all raw columns plus Zone assignment and USGS-LR outputs.
"""

import math
import openpyxl
from config import (
    DATA_FILE, SHEET_NAME,
    INTERCEPT, COEFF_X1, COEFF_X2, COEFF_X3,
    FIRE_TO_ZONE,
)


def assign_zone(fire_name: str) -> str:
    zone = FIRE_TO_ZONE.get(fire_name)
    if zone is None:
        raise ValueError(f"Unknown fire name: {fire_name!r}")
    return zone


def _compute_x(X1, X2, X3, R) -> float:
    return INTERCEPT + (COEFF_X1 * X1 * R) + (COEFF_X2 * X2 * R) + (COEFF_X3 * X3 * R)


def _sigmoid(x: float) -> float:
    ex = math.exp(x)
    return ex / (1 + ex)


def load_data(filepath=None) -> list:
    """
    Load Excel → list of event dicts (1,243 events).

    Each event dict contains raw columns, Zone assignment,
    USGS logit x, and USGS probability P.
    Events missing Response or USGS features (R, X1, X2, X3) are skipped.
    """
    path = filepath or DATA_FILE
    wb = openpyxl.load_workbook(path)
    sheet = wb[SHEET_NAME]

    rows = sheet.iter_rows(values_only=True)
    headers = next(rows)
    col = {name: i for i, name in enumerate(headers)}

    events = []
    for row in rows:
        if row[col["Response"]] is None:
            continue

        R  = row[col["Acc015_mm"]]
        X1 = row[col["PropHM23"]]
        X2 = row[col["dNBR/1000"]]
        X3 = row[col["KF"]]

        if any(v is None or isinstance(v, str) for v in (R, X1, X2, X3)):
            continue

        x = _compute_x(X1, X2, X3, R)
        P = _sigmoid(x)
        fire_name = row[col["Fire Name"]]

        events.append({
            "Fire_SegID":      row[col["Fire_SegID"]],
            "Fire_Name":       fire_name,
            "Fire_ID":         row[col["Fire_ID"]],
            "Year":            row[col["Year"]],
            "Database":        row[col["Database"]],
            "State":           row[col["State"]],
            "UTM_Zone":        row[col["UTM_Zone"]],
            "UTM_X":           row[col["UTM_X"]],
            "UTM_Y":           row[col["UTM_Y"]],
            "Zone":            assign_zone(fire_name),
            "Response":        int(row[col["Response"]]),
            "StormDate":       row[col["StormDate"]],
            "StormStart":      row[col["StormStart"]],
            "StormEnd":        row[col["StormEnd"]],
            "StormDur_H":      row[col["StormDur_H"]],
            "StormAccum_mm":   row[col["StormAccum_mm"]],
            "StormAvgI_mmph":  row[col["StormAvgI_mm/h"]],
            "Peak_I15_mmph":   row[col["Peak_I15_mm/h"]],
            "Peak_I30_mmph":   row[col["Peak_I30_mm/h"]],
            "Peak_I60_mmph":   row[col["Peak_I60_mm/h"]],
            "ContribArea_km2": row[col["ContributingArea_km2"]],
            "GaugeDist_m":     row[col["GaugeDist_m"]],
            "Acc015_mm":       row[col["Acc015_mm"]],
            "Acc030_mm":       row[col["Acc030_mm"]],
            "Acc060_mm":       row[col["Acc060_mm"]],
            "R":               R,
            "X1":              X1,
            "X2":              X2,
            "X3":              X3,
            "x":               x,
            "P":               P,
        })

    return events
