"""
features/split.py — Train / val / test split strategies.

All splits operate on a `paired` list of (event_dict, aux) tuples where
event_dict contains "Response" (int) and "Zone" (str).
Returns index lists into that paired list.
"""

import random
from collections import defaultdict
from config import ZONE_ORDER


def zone_stratified_split_3way(paired, train_ratio, val_ratio, seed=42):
    """
    Stratified 3-way split by (zone, label).

    Each (zone, label) stratum is shuffled and split proportionally into
    train / val / test. Very small strata may not appear in every split after
    rounding; this is expected for rare positive zones in the complete-case
    dataset.

    Returns
    -------
    train_idx, val_idx, test_idx : lists of indices into `paired`
    """
    rng = random.Random(seed)

    strata = defaultdict(list)
    for i, (event, _) in enumerate(paired):
        key = (event["Zone"], int(event["Response"]))
        strata[key].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for indices in strata.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_tr = max(1, round(n * train_ratio))
        n_va = max(1, round(n * val_ratio))
        train_idx.extend(shuffled[:n_tr])
        val_idx.extend(shuffled[n_tr:n_tr + n_va])
        test_idx.extend(shuffled[n_tr + n_va:])

    return train_idx, val_idx, test_idx


def original_split(paired):
    """
    Original USGS geographic split: Database == 'Training' → train,
    else → test.  No validation set.

    Returns train_idx, test_idx.
    """
    train_idx, test_idx = [], []
    for i, (event, _) in enumerate(paired):
        if event.get("Database") == "Training":
            train_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, test_idx


def grouped_stratified_split_3way(
    paired,
    train_ratio,
    val_ratio,
    group_fn,
    seed=42,
):
    """
    Group-aware 3-way split that keeps whole groups together.

    The assignment is greedy rather than exact: groups are shuffled, then placed
    into train / val / test to approximately match target split sizes and target
    positive-label counts. This is intended for leakage-sensitive studies such as
    repeated weather-document bundles.

    Parameters
    ----------
    paired    : list of (event_dict, aux)
    group_fn  : callable(event_dict, aux) -> hashable group key

    Returns
    -------
    train_idx, val_idx, test_idx : lists of indices into `paired`
    """
    rng = random.Random(seed)

    group_to_indices = defaultdict(list)
    for i, (event, aux) in enumerate(paired):
        group_to_indices[group_fn(event, aux)].append(i)

    groups = []
    n_total = len(paired)
    y_total = sum(int(event["Response"]) for event, _ in paired)
    targets = {
        "train": {"n": train_ratio * n_total, "y": train_ratio * y_total},
        "val":   {"n": val_ratio * n_total,   "y": val_ratio * y_total},
        "test":  {"n": (1.0 - train_ratio - val_ratio) * n_total,
                  "y": (1.0 - train_ratio - val_ratio) * y_total},
    }
    assigned = {
        "train": {"idx": [], "n": 0, "y": 0},
        "val":   {"idx": [], "n": 0, "y": 0},
        "test":  {"idx": [], "n": 0, "y": 0},
    }

    for group_key, idxs in group_to_indices.items():
        group_y = sum(int(paired[i][0]["Response"]) for i in idxs)
        groups.append((group_key, idxs, len(idxs), group_y))

    rng.shuffle(groups)
    groups.sort(key=lambda item: (item[2], item[3]), reverse=True)

    split_names = ["train", "val", "test"]
    for _, idxs, group_n, group_y in groups:
        best_split = None
        best_score = None
        for split_name in split_names:
            cur = assigned[split_name]
            tgt = targets[split_name]
            new_n = cur["n"] + group_n
            new_y = cur["y"] + group_y
            size_err = abs(new_n - tgt["n"]) / max(tgt["n"], 1.0)
            pos_err = abs(new_y - tgt["y"]) / max(tgt["y"], 1.0)
            score = size_err + pos_err
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        assigned[best_split]["idx"].extend(idxs)
        assigned[best_split]["n"] += group_n
        assigned[best_split]["y"] += group_y

    return (
        assigned["train"]["idx"],
        assigned["val"]["idx"],
        assigned["test"]["idx"],
    )
