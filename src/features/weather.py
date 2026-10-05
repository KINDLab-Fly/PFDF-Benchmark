"""
features/weather.py — Load NWS forecast text features.

Representations:
  load_weather_embeddings(events)                → (N, 384) AFD embeddings
  load_weather_embeddings_multi(events, fields)  → (N, 384*len(fields)) concatenated embeddings
  load_weather_onehot(events)                    → (N, K) binary one-hot over unique AFD source IDs

Used by NW, richer weather-text variants, and grouping-effect ablations.
Coverage: AFD 96.6%, HWO 76.1%, FFW/FFA 59.2%.  Missing → zero vector.
"""

import json
import torch
import pandas as pd
from pathlib import Path

NWS_TEXTS_FILE  = "data/nws_texts.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def weather_lookup_key(event) -> str:
    """Build nws_texts.json lookup key:  'Fire_SegID|storm_date'."""
    seg_id = str(event.get("Fire_SegID", ""))
    storm  = event.get("StormStart") or event.get("StormDate")
    if storm is None:
        return ""
    if not hasattr(storm, "date"):
        storm = pd.to_datetime(storm)
    return f"{seg_id}|{storm.date()}"


def load_nws_cache() -> dict:
    if not Path(NWS_TEXTS_FILE).exists():
        raise FileNotFoundError(f"'{NWS_TEXTS_FILE}' not found. Run data_collection/fetch_afd.py first.")
    with open(NWS_TEXTS_FILE) as f:
        return json.load(f)


def weather_group_key(event, nws_cache: dict | None = None):
    """
    Group key for leakage-sensitive weather experiments.

    Uses the tuple of source-document identifiers rather than the text itself so
    repeated AFD/HWO/FFW bundles stay together in grouped splits.
    """
    nws = nws_cache if nws_cache is not None else load_nws_cache()
    entry = nws.get(weather_lookup_key(event), {})
    return (
        entry.get("afd_src") or "",
        entry.get("hwo_src") or "",
        entry.get("ffw_src") or "",
    )


def load_weather_embeddings(events: list) -> torch.Tensor:
    """
    Load AFD text for each event and encode with sentence-transformers.

    Returns (N, 384) float32 tensor.  Events without an AFD entry get zero rows.
    """
    from sentence_transformers import SentenceTransformer

    nws = load_nws_cache()

    texts = [nws.get(weather_lookup_key(e), {}).get("afd", "") for e in events]

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    embs = encoder.encode(texts, show_progress_bar=True, convert_to_tensor=True).float().cpu()
    return embs   # (N, 384)


def load_weather_embeddings_multi(events: list, fields=("afd", "hwo", "ffw")) -> torch.Tensor:
    """
    Encode multiple NWS text products and concatenate the resulting embeddings.

    Parameters
    ----------
    fields : iterable of product keys from nws_texts.json, e.g. ("afd", "hwo", "ffw")

    Returns
    -------
    (N, 384 * len(fields)) float32 tensor
    """
    from sentence_transformers import SentenceTransformer

    nws = load_nws_cache()
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    parts = []
    for field in fields:
        texts = [nws.get(weather_lookup_key(e), {}).get(field, "") for e in events]
        embs = encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_tensor=True,
        ).float().cpu()
        parts.append(embs)
    return torch.cat(parts, dim=1)


def load_weather_onehot(events: list) -> torch.Tensor:
    """
    Build a binary one-hot over unique AFD source IDs (afd_src field).

    Each unique NWS product ID gets one dimension.  Events with no matching
    AFD source get an all-zero row.

    Returns (N, K) float32 tensor where K = number of unique sources.
    """
    nws = load_nws_cache()

    srcs = []
    for e in events:
        entry = nws.get(weather_lookup_key(e), {})
        srcs.append(entry.get("afd_src") or None)

    vocab  = sorted(set(s for s in srcs if s))
    src2i  = {s: i for i, s in enumerate(vocab)}
    n_srcs = len(vocab)

    oh = torch.zeros(len(events), n_srcs)
    for i, src in enumerate(srcs):
        if src and src in src2i:
            oh[i, src2i[src]] = 1.0

    n_covered = sum(1 for s in srcs if s)
    print(f"  AFD one-hot: {n_srcs} unique sources, "
          f"{n_covered}/{len(events)} events covered ({100*n_covered/len(events):.1f}%)")
    return oh   # (N, K)
