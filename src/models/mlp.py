"""
models/mlp.py — Plain MLP with learnable categorical embeddings.

EmbeddingMLP replaces fixed encodings (ordinal, one-hot) for categorical
features with trainable nn.Embedding tables.  All other features are
treated as z-scored continuous inputs.

Design
------
Each categorical variable is specified as a CatSpec(col, n_cats, emb_dim):
  col      : column index in the input tensor (integer-valued float)
  n_cats   : number of categories
  emb_dim  : embedding dimension (learnable)

The model:
  1. Extracts each categorical column, converts to long, looks up embedding.
  2. Removes categorical columns from the continuous part.
  3. Concatenates continuous features + all embeddings.
  4. Passes through a 3-layer MLP: in→h→h//2→1.

Default configuration (N and N_aug models):
  - Zone ordinal at column 13, 10 categories, 8-dim embedding.
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class CatSpec:
    col: int       # column index in input tensor
    n_cats: int    # number of categories
    emb_dim: int   # embedding dimension


# Default zone spec — used by N and N_aug
ZONE_N_CATS  = 10
ZONE_EMB_DIM = 8
ZONE_COL     = 13   # index of zone_ord in both 14-dim (N) and 19-dim (N_aug) tensors

DEFAULT_CAT_SPECS = [CatSpec(col=ZONE_COL, n_cats=ZONE_N_CATS, emb_dim=ZONE_EMB_DIM)]


class EmbeddingMLP(nn.Module):
    """
    Plain MLP with learnable categorical embeddings.

    Parameters
    ----------
    feat_dim   : total number of input columns (including categorical columns)
    hidden     : hidden layer size
    dropout    : dropout probability
    cat_specs  : list of CatSpec for each categorical feature
                 (default: [zone at col 13, 10 cats, 8-dim])
    """

    def __init__(self, feat_dim: int, hidden: int, dropout: float,
                 cat_specs: List[CatSpec] = None):
        super().__init__()
        self.cat_specs = cat_specs if cat_specs is not None else DEFAULT_CAT_SPECS

        self.embeddings = nn.ModuleList([
            nn.Embedding(s.n_cats, s.emb_dim) for s in self.cat_specs
        ])

        cat_cols  = {s.col for s in self.cat_specs}
        n_cont    = feat_dim - len(cat_cols)
        n_emb_dim = sum(s.emb_dim for s in self.cat_specs)
        in_dim    = n_cont + n_emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

        self._cat_cols_sorted = sorted(cat_cols)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed each categorical column
        emb_parts = [
            emb(x[:, s.col].long())
            for emb, s in zip(self.embeddings, self.cat_specs)
        ]

        # Remove categorical columns from continuous part
        all_cols   = list(range(x.shape[1]))
        cont_cols  = [c for c in all_cols if c not in self._cat_cols_sorted]
        x_cont     = x[:, cont_cols]

        # Concatenate and pass through MLP
        parts = [x_cont] + emb_parts
        return self.mlp(torch.cat(parts, dim=1)).squeeze(-1)
