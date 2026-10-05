"""
experiments/logging_utils.py — Shared experiment run logging helpers.

Each experiment creates a timestamped run directory under:
    results/research_runs/<run_id>/

Artifacts:
  - metadata.json      : static run metadata and environment snapshot
  - events.jsonl       : append-only stage/event log
  - summary.json       : final status, outputs, and notes
  - notes.md           : human-readable execution notes written by the script

The goal is to keep a documented audit trail for every major research step.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch


RESULTS_ROOT = Path("results/research_runs")
INDEX_FILE = RESULTS_ROOT / "index.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _safe_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


def _query_nvidia_smi() -> Optional[Dict[str, str]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception:
        return None

    if not out:
        return None
    parts = [p.strip() for p in out.split(",")]
    if len(parts) != 3:
        return {"raw": out}
    return {
        "gpu_name": parts[0],
        "memory_total": parts[1],
        "driver_version": parts[2],
    }


def get_environment_snapshot() -> Dict[str, Any]:
    snapshot = {
        "timestamp_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "python": os.sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        snapshot["cuda_device_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    nvidia = _query_nvidia_smi()
    if nvidia:
        snapshot["nvidia_smi"] = nvidia
    return snapshot


@dataclass
class RunLogger:
    experiment_name: str
    run_id: str
    run_dir: Path
    metadata_path: Path
    events_path: Path
    summary_path: Path
    notes_path: Path
    artifacts: list = field(default_factory=list)

    def log_event(self, stage: str, message: str, **payload: Any) -> None:
        event = {
            "timestamp_utc": _utc_now(),
            "stage": stage,
            "message": message,
            "payload": _safe_json(payload),
        }
        with self.events_path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    def log_artifact(self, path: str | Path, kind: str, description: str) -> None:
        record = {
            "path": str(path),
            "kind": kind,
            "description": description,
        }
        self.artifacts.append(record)
        self.log_event("artifact", f"Recorded {kind} artifact", artifact=record)

    def write_notes(self, text: str) -> None:
        self.notes_path.write_text(text)
        self.log_event("notes", "Updated human-readable notes", notes_file=self.notes_path)

    def finalize(self, status: str, summary: Dict[str, Any]) -> None:
        payload = {
            "timestamp_utc": _utc_now(),
            "status": status,
            "artifacts": self.artifacts,
            "summary": _safe_json(summary),
        }
        self.summary_path.write_text(json.dumps(payload, indent=2))
        self.log_event("finalize", f"Run finished with status={status}", summary=summary)


def create_run_logger(
    experiment_name: str,
    config: Dict[str, Any],
    description: str,
) -> RunLogger:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    base_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slugify(experiment_name)}"
    run_id = base_id
    run_dir = RESULTS_ROOT / run_id
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_id = f"{base_id}_{suffix:02d}"
        run_dir = RESULTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "description": description,
        "config": _safe_json(config),
        "environment": get_environment_snapshot(),
    }

    metadata_path = run_dir / "metadata.json"
    events_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"
    notes_path = run_dir / "notes.md"

    metadata_path.write_text(json.dumps(metadata, indent=2))
    with INDEX_FILE.open("a") as f:
        f.write(json.dumps({
            "run_id": run_id,
            "experiment_name": experiment_name,
            "timestamp_utc": metadata["environment"]["timestamp_utc"],
            "run_dir": str(run_dir),
        }) + "\n")

    logger = RunLogger(
        experiment_name=experiment_name,
        run_id=run_id,
        run_dir=run_dir,
        metadata_path=metadata_path,
        events_path=events_path,
        summary_path=summary_path,
        notes_path=notes_path,
    )
    logger.log_event("start", "Created experiment run logger", config=config)
    return logger
