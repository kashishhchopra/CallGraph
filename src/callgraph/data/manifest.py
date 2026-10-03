"""Manifest CSV helpers (columns come from ``settings.yaml``)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from callgraph.config import AppConfig


def manifest_path(config: AppConfig, dataset: str) -> Path:
    """Path of the manifest CSV for a dataset."""
    return config.path("manifests_dir") / f"{dataset}.csv"


def write_manifest(config: AppConfig, dataset: str, rows: list[dict[str, Any]]) -> Path:
    """Write rows to ``data/manifests/<dataset>.csv`` with the configured columns."""
    cols = list(config.manifest.columns)  # type: ignore[attr-defined]
    frame = pd.DataFrame(rows)
    for col in cols:
        if col not in frame:
            frame[col] = None
    frame["labels"] = frame["labels"].apply(
        lambda v: v if isinstance(v, str) or v is None else json.dumps(v, ensure_ascii=False)
    )
    path = manifest_path(config, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[cols].to_csv(path, index=False)
    return path


def read_manifest(config: AppConfig, dataset: str) -> pd.DataFrame:
    """Read a manifest; ``labels`` is decoded from JSON where possible."""
    frame = pd.read_csv(manifest_path(config, dataset))

    def _dec(v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v

    frame["labels"] = frame["labels"].apply(_dec)
    return frame


def assign_group_splits(groups: list[str], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Deterministic group-level split (no speaker/dialogue leakage)."""
    names = list(ratios)
    out: dict[str, str] = {}
    for g in groups:
        h = random.Random(f"{seed}:{g}").random()
        acc = 0.0
        for n in names:
            acc += ratios[n]
            if h < acc:
                out[g] = n
                break
        else:
            out[g] = names[-1]
    return out
