#!/usr/bin/env python3
"""Evaluate fitted research checkpoints and export portable held-out predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiments.reproduce_checkpointed_baselines import _load_complete_case, _metrics, _score_model, _slug
from experiments.eval_literature_baselines import _build_feature_matrix
from models.mlp import EmbeddingMLP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/checkpoint-evaluation")
    args = parser.parse_args()
    # Resolve before research imports change cwd; accept only explicit local artifacts.
    checkpoints = args.checkpoint_root.resolve()
    records_dir = args.records_dir.resolve()
    output = args.output_dir.resolve()
    torch.set_num_threads(1)
    df, _, weather = _load_complete_case()
    arrays = {"labels": df.pfdf.to_numpy(dtype=np.int32), "zones": df.zone.to_numpy(dtype=str), "usgs_scores": df.usgs_p.to_numpy()}
    exported, differences = [], []
    cache = {}
    for config in ["original", "augmented"]:
        record_path = records_dir / f"historical-{config}-records.json"
        records = json.loads(record_path.read_text())
        for i, rec in enumerate(records):
            ratio, seed, model = rec["ratio_label"], int(rec["split_seed"]), rec["model"]
            key = (config, ratio, seed)
            if key not in cache:
                X = _build_feature_matrix("n" if config == "original" else "selected_aug",
                                          df, rec["train_idx"], weather)
                cache[key] = X
            X = cache[key]
            te = np.asarray(rec["test_idx"], dtype=np.int32)
            directory = checkpoints / config / _slug(ratio) / f"split_seed_{seed}"
            eval_seed = rec.get("eval_seed")
            if eval_seed is None:
                with (directory / (model + ".pkl")).open("rb") as handle:
                    fitted = pickle.load(handle)["model"]
                scores = _score_model(fitted, X[te])
            else:
                path = directory / f"{model}_eval_seed_{eval_seed}.pt"
                payload = torch.load(path, map_location="cpu", weights_only=False)
                hp = payload["metadata"]["best_hp"]
                fitted = EmbeddingMLP(X.shape[1], hp["hidden"], hp["dropout"])
                fitted.load_state_dict(payload["state_dict"])
                fitted.eval()
                with torch.no_grad():
                    scores = torch.sigmoid(fitted(torch.tensor(X[te], dtype=torch.float32))).numpy()
            metrics = _metrics(df.pfdf.to_numpy()[te], scores)
            delta = metrics["TS"] - rec["test_metrics"]["TS"]
            prefix = f"{config}_{_slug(ratio)}_split{seed}_{model}_init{eval_seed}"
            arrays[prefix + "_indices"] = te
            arrays[prefix + "_scores"] = scores
            item = {"config": config, "model": model, "ratio": ratio, "split_seed": seed,
                    "eval_seed": eval_seed, "array_prefix": prefix, "TS": metrics["TS"],
                    "historical_TS": rec["test_metrics"]["TS"]}
            exported.append(item)
            if abs(delta) > 1e-10:
                differences.append(item)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "heldout-predictions.npz", **arrays)
    report = {"evaluated_checkpoints": len(exported), "matches_historical_record": len(exported)-len(differences),
              "differences": differences, "records": exported,
              "scope": "Inference using historical trained checkpoints on canonical held-out folds; no retraining."}
    (output / "prediction-records.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Checkpoints evaluated:", len(exported), "differing metrics:", len(differences))


if __name__ == "__main__":
    main()
