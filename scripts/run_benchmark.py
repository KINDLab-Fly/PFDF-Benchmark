#!/usr/bin/env python3
"""Offline benchmark entrypoint: saved-HP refits, full searches, and RL reruns."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "refit", "search", "rl"], default="smoke")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[key] = str(args.threads)
    env["MPLBACKEND"] = "Agg"

    def run(module, *options):
        command = [sys.executable, "-m", module, *options]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    if args.mode in {"smoke", "refit"}:
        for config in ["original", "augmented"]:
            options = ["--config", config]
            if args.mode == "smoke":
                options += ["--ratios", "70/15/15", "--split-seeds", "42", "--eval-seeds", "42"]
            run("experiments.reproduce_checkpointed_baselines", *options)
    elif args.mode == "search":
        # Independent validation-selected grid searches, including both neural variants.
        for features in ["n", "selected_aug"]:
            run("experiments.eval_literature_baselines", "--feature-set", features,
                "--output-tag", "release-rerun-" + features)
        run("experiments.eval_main", "--models", "N", "--output-tag", "release-rerun-original-nn")
        run("experiments.eval_n_targeted_combo_campaign", "--models", "N+TargetedCombo",
            "--output-tag", "release-rerun-augmented-nn")
    else:
        # Requires augmented refit checkpoints for every split seed at 70/15/15.
        for seed in range(42, 47):
            run("experiments.eval_rl_feature_selection_from_checkpoints", "--config", "augmented",
                "--ratio", "70/15/15", "--split-seed", str(seed), "--resume")
        run("experiments.eval_rl_feature_selection_from_checkpoints", "--mode", "aggregate",
            "--config", "augmented", "--ratio", "70/15/15")


if __name__ == "__main__":
    main()
