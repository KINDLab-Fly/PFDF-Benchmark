"""
models/graph_mlp.py — Graph-constrained MLP (G*-MLP) with learnable zone embedding.

Architecture
------------
1. Node embedding: each input feature scalar → h-dim vector via nn.Linear(1, h).
   Exception: zone_ord node uses nn.Embedding(10, h) — learnable per-zone representation.
2. Graph message passing (1 layer):
   Each node j aggregates hidden states of its G* parents:
     agg[j] = sum_{i: i→j in G*} msg_weight(h[i])
   Node state updated: h2[j] = ReLU(Linear([h[j]; agg[j]]))
3. Readout: flatten all node states → MLP → scalar logit.

G* edges are the 32 edges from the refined causal graph (PC + LiNGAM + LLM priors,
composite score ≥ 0.20).  gauge_dist is isolated → direct edge gauge_dist→pfdf added.
"""

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Causal graph G* (32 edges, hardcoded from causal_graph_refined.csv)
# ─────────────────────────────────────────────────────────────────────────────

G_STAR_EDGES = [
    # LLM + Data confirmed (composite ≥ 0.20)
    ("peak_i15",        "acc030"),
    ("soilm_7d",        "soilm_30d"),
    ("storm_accum",     "acc060"),
    ("storm_accum",     "acc030"),
    ("storm_dur",       "storm_avg_i"),
    ("storm_accum",     "pfdf"),
    ("x1",              "pfdf"),
    # LLM-only (score ≥ 0.20)
    ("x2",              "x1"),
    ("slope_deg",       "x1"),
    ("storm_accum",     "storm_avg_i"),
    ("x2",              "soilm_7d"),
    ("storm_accum",     "soilm_7d"),
    ("zone_ord",        "soilm_7d"),
    ("time_since_fire", "soilm_7d"),
    ("acc030",          "pfdf"),
    ("acc060",          "pfdf"),
    ("peak_i15",        "pfdf"),
    ("storm_avg_i",     "pfdf"),
    # Data-only, physically plausible
    ("storm_dur",       "storm_accum"),
    ("acc030",          "acc060"),
    ("month_cos",       "peak_i15"),
    ("month_cos",       "storm_dur"),
    ("month_cos",       "storm_accum"),
    ("month_sin",       "storm_accum"),
    ("month_cos",       "acc060"),
    ("peak_i15",        "acc060"),
    ("storm_avg_i",     "peak_i15"),
    ("month_sin",       "acc030"),
    ("month_cos",       "acc030"),
    ("storm_avg_i",     "acc030"),
    # gauge_dist: isolated in G* — direct path to pfdf
    ("gauge_dist",      "pfdf"),
]

ZONE_N_CATS = 10   # OSD, SGSBSJ, VEN, NAZ, SAZ, CNM, FRCO, MT, SWCO, WCO


class GraphMLP(nn.Module):
    """
    G*-masked MLP.  All nodes use a shared Linear(1→h) for embedding.
    This is the base class; EmbeddingGraphMLP overrides zone_ord.
    """

    def __init__(self, n_features: int, feature_names: list,
                 hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.n            = n_features
        self.feature_names = feature_names
        self.hidden       = hidden

        self.node_embed   = nn.Linear(1, hidden)

        # Build G* adjacency mask  A[i,j]=1 means i→j (i is parent of j)
        feat_idx = {name: i for i, name in enumerate(feature_names)}
        A = torch.zeros(n_features, n_features)
        for src, tgt in G_STAR_EDGES:
            if src in feat_idx and tgt in feat_idx:
                A[feat_idx[src], feat_idx[tgt]] = 1.0
        self.register_buffer("adj", A)

        self.msg_weight  = nn.Linear(hidden, hidden, bias=False)
        self.node_update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden * n_features, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_features)"""
        B, N = x.shape
        h    = self.node_embed(x.unsqueeze(-1))            # (B, N, h)
        msgs = self.msg_weight(h)
        agg  = torch.einsum("bih,ij->bjh", msgs, self.adj) # (B, N, h)
        h2   = self.node_update(torch.cat([h, agg], dim=-1))
        return self.readout(h2.view(B, -1)).squeeze(-1)


class EmbeddingGraphMLP(GraphMLP):
    """
    G*-MLP with a learnable zone embedding.

    All nodes: nn.Linear(1, hidden).
    zone_ord node: nn.Embedding(ZONE_N_CATS, hidden) — each of the 10 zones
    gets its own independent hidden-dim representation, replacing the
    arbitrary ordinal scalar encoding.
    """

    def __init__(self, n_features: int, feature_names: list,
                 hidden: int = 64, dropout: float = 0.3):
        super().__init__(n_features, feature_names, hidden, dropout)
        self.zone_col   = feature_names.index("zone_ord")
        self.zone_embed = nn.Embedding(ZONE_N_CATS, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape

        # Standard linear embedding for all nodes
        h = self.node_embed(x.unsqueeze(-1))               # (B, N, h)

        # Replace zone_ord node's state with learnable embedding
        h[:, self.zone_col, :] = self.zone_embed(x[:, self.zone_col].long())

        # Message passing (unchanged)
        msgs = self.msg_weight(h)
        agg  = torch.einsum("bih,ij->bjh", msgs, self.adj)
        h2   = self.node_update(torch.cat([h, agg], dim=-1))
        return self.readout(h2.view(B, -1)).squeeze(-1)
