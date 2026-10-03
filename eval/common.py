"""Shared helpers for evaluation runners: run directories, saving tables/figures, manifests."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from callgraph.config import AppConfig  # noqa: E402
from callgraph.data.manifest import read_manifest  # noqa: E402

RUN_ENV = "CALLGRAPH_RUN_ID"


def run_dir(config: AppConfig, run_id: str | None = None) -> Path:
    """``outputs/eval/<run_id>`` (run id from arg, env, or newest existing, else new timestamp)."""
    base = config.path("outputs_dir") / "eval"
    rid = run_id or os.environ.get(RUN_ENV)
    if not rid:
        existing = sorted(p.name for p in base.glob(f"{config.evaluation.run_id_prefix}_*")) if base.exists() else []  # type: ignore[attr-defined]
        rid = existing[-1] if existing else f"{config.evaluation.run_id_prefix}_{datetime.now():%Y%m%d_%H%M%S}"  # type: ignore[attr-defined]
    d = base / rid
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_result(
    config: AppConfig, name: str, data: dict[str, Any], table: pd.DataFrame | None = None
) -> Path:
    """Persist ``<name>.json`` (+ ``<name>.csv``) in the run directory."""
    d = run_dir(config)
    (d / f"{name}.json").write_text(json.dumps(data, indent=2, default=str))
    if table is not None:
        table.to_csv(d / f"{name}.csv", index=False)
    return d / f"{name}.json"


def save_fig(config: AppConfig, name: str) -> Path:
    path = run_dir(config) / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=int(config.evaluation.plots["dpi"]))  # type: ignore[attr-defined]
    plt.close()
    return path


def subset(config: AppConfig, dataset: str, n: int, split: str | None = None) -> pd.DataFrame:
    """First ``n`` manifest rows (optionally of a split)."""
    man = read_manifest(config, dataset)
    if split:
        man = man[man["split"] == split]
    return man.head(n)


def load_doc(config: AppConfig, row: pd.Series) -> dict[str, Any]:
    """Transcript JSON referenced by a manifest row."""
    return json.loads((config.root / row["transcript_path"]).read_text())
