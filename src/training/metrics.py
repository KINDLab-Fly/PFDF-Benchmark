"""
training/metrics.py — Evaluation metrics and reproducibility utilities.
"""

import random
import numpy as np
import torch


def threat_score(logits: torch.Tensor, labels: torch.Tensor,
                 threshold: float = 0.5) -> float:
    """
    Threat Score (Critical Success Index) = TP / (TP + FP + FN).

    logits : raw model output (pre-sigmoid), shape (N,)
    labels : binary ground truth, shape (N,)
    """
    preds = (torch.sigmoid(logits) >= threshold).long()
    y     = labels.long()
    tp    = int(((preds == 1) & (y == 1)).sum())
    fp    = int(((preds == 1) & (y == 0)).sum())
    fn    = int(((preds == 0) & (y == 1)).sum())
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def set_seed(seed: int) -> None:
    """Set seeds for numpy, Python random, and PyTorch (CPU + GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
