"""
training/trainer.py — Full-batch training loop with early stopping.

Uses GPU when available.  All models and tensors are moved to DEVICE
inside train_one(); callers pass CPU tensors.
"""

import copy
import torch
import torch.nn as nn

from .metrics import threat_score, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one(model_fn, X_tr, y_tr, X_va, y_va, X_te, y_te,
              hp: dict, seed: int,
              max_epochs: int = 400, patience: int = 50):
    """
    Train one model instance using full-batch gradient descent.

    Parameters
    ----------
    model_fn   : callable(hp) → nn.Module
    X_tr/va/te : float32 tensors (CPU); moved to DEVICE internally
    y_tr/va/te : float32 label tensors (CPU)
    hp         : dict with keys lr, dropout, hidden, weight_decay
    seed       : controls weight initialisation
    max_epochs : hard epoch limit
    patience   : early-stopping patience on val Threat Score

    Returns
    -------
    (val_ts, test_ts) : best validation TS and its corresponding test TS
    """
    set_seed(seed)
    model = model_fn(hp).to(DEVICE)

    X_tr, y_tr = X_tr.to(DEVICE), y_tr.to(DEVICE)
    X_va, y_va = X_va.to(DEVICE), y_va.to(DEVICE)
    X_te, y_te = X_te.to(DEVICE), y_te.to(DEVICE)

    n_pos = max(float(y_tr.sum()), 1.0)
    n_neg = float(len(y_tr)) - n_pos
    crit  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_neg / n_pos], device=DEVICE)
    )
    opt = torch.optim.Adam(
        model.parameters(),
        lr=hp["lr"],
        weight_decay=hp.get("weight_decay", 0.0),
    )

    best_val   = -1.0
    best_test  = float("nan")
    best_state = None
    no_imp     = 0

    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        crit(model(X_tr), y_tr).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            v_ts = threat_score(model(X_va), y_va)
            t_ts = threat_score(model(X_te), y_te)

        if v_ts > best_val:
            best_val   = v_ts
            best_test  = t_ts
            best_state = copy.deepcopy(model.state_dict())
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_val, best_test
