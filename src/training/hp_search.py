"""
training/hp_search.py — Unified hyperparameter search protocol.

All experiments use exactly these settings for a fair comparison.

Protocol
--------
HP grid     : 200 configs  (4 lr × 5 dropout × 5 hidden × 2 weight_decay)
HP selection: best mean val TS over HP_SEEDS (5 weight-init seeds)
              on a FIXED split (SPLIT_SEED=42)
Final eval  : EVAL_SEEDS (10 weight-init seeds) on the same fixed split
Training    : full-batch gradient descent, Adam, BCEWithLogitsLoss + pos_weight
              early stopping (patience=PATIENCE) on val Threat Score
"""

import numpy as np
from features.split import zone_stratified_split_3way
from .trainer import train_one

# ── Fixed experimental protocol ───────────────────────────────────────────────

SPLIT_RATIOS = [
    (0.50, 0.25, "50/25/25"),
    (0.60, 0.20, "60/20/20"),
    (0.70, 0.15, "70/15/15"),
    (0.80, 0.10, "80/10/10"),
]

MAX_EPOCHS = 400
PATIENCE   = 50
SPLIT_SEED = 42   # fixed data partition — only weight init varies across seeds

HP_SEEDS   = [42, 43, 44, 45, 46]        # 5 seeds for HP selection
EVAL_SEEDS = list(range(42, 52))          # 10 seeds for final evaluation

HP_GRID = [
    {"lr": lr, "dropout": dr, "hidden": hd, "weight_decay": wd}
    for lr in [1e-4, 3e-4, 1e-3, 3e-3]
    for dr in [0.0, 0.1, 0.2, 0.3, 0.5]
    for hd in [32, 64, 128, 256, 512]
    for wd in [0.0, 1e-4]
]
# 4 × 5 × 5 × 2 = 200 configs


# ── HP search + final evaluation ──────────────────────────────────────────────

def run_one(get_splits, model_fn_factory, paired,
            train_ratio: float, val_ratio: float,
            split_fn=None,
            split_seed: int = SPLIT_SEED) -> dict:
    """
    Run HP search then final evaluation for one (model, split-ratio) combination.

    Parameters
    ----------
    get_splits       : callable(tr, va, te) → (X_tr, y_tr, X_va, y_va, X_te, y_te)
    model_fn_factory : callable(feat_dim) → callable(hp) → nn.Module
    paired           : list of (event_dict, aux) for zone_stratified_split_3way
    train_ratio      : fraction of data for training
    val_ratio        : fraction for validation (remainder → test)

    Returns
    -------
    dict with keys: best_hp, test_ts (list), mean_test, std_test
    """
    # Fixed split — same partition for all seeds
    split_strategy = split_fn or zone_stratified_split_3way
    tr, va, te = split_strategy(
        paired, train_ratio=train_ratio, val_ratio=val_ratio, seed=split_seed
    )
    X_tr, y_tr, X_va, y_va, X_te, y_te = get_splits(tr, va, te)
    feat_dim = X_tr.shape[1]

    # HP search: average val TS over HP_SEEDS
    best_hp  = HP_GRID[0]
    best_val = -1.0

    for hp in HP_GRID:
        vals = []
        for seed in HP_SEEDS:
            mfn  = model_fn_factory(feat_dim)
            v_ts, _ = train_one(mfn, X_tr, y_tr, X_va, y_va, X_te, y_te,
                                hp, seed, MAX_EPOCHS, PATIENCE)
            vals.append(v_ts)
        mean_v = float(np.mean(vals))
        if mean_v > best_val:
            best_val = mean_v
            best_hp  = hp

    # Final eval: EVAL_SEEDS weight inits, same fixed split
    test_ts_list = []
    for seed in EVAL_SEEDS:
        mfn  = model_fn_factory(feat_dim)
        _, t_ts = train_one(mfn, X_tr, y_tr, X_va, y_va, X_te, y_te,
                            best_hp, seed, MAX_EPOCHS, PATIENCE)
        test_ts_list.append(t_ts)

    return {
        "split_seed": split_seed,
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "best_hp":   best_hp,
        "test_ts":   test_ts_list,
        "mean_test": float(np.mean(test_ts_list)),
        "std_test":  float(np.std(test_ts_list)),
    }
