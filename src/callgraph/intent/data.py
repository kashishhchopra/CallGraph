"""Intent training/eval data built from manifests (shared label set)."""

from __future__ import annotations

import json
import random
from typing import Any

import pandas as pd

from callgraph.config import AppConfig
from callgraph.data.manifest import read_manifest


def customer_text(
    turns: list[dict[str, Any]], k: int | None = None, until_s: float | None = None
) -> str:
    """Join customer turns; limit to the first ``k`` turns or those starting before ``until_s``."""
    cust = [t for t in turns if t.get("role") == "customer"]
    if until_s is not None:
        cust = [t for t in cust if t.get("start", 0.0) < until_s]
    if k is not None:
        cust = cust[:k]
    return " ".join(t["text"] for t in cust)


def load_frame(config: AppConfig, dataset: str, splits: list[str] | None = None) -> pd.DataFrame:
    """Return (id, split, turns, intent) rows for a dataset manifest."""
    man = read_manifest(config, dataset)
    if splits:
        man = man[man["split"].isin(splits)]
    rows = []
    for _, r in man.iterrows():
        doc = json.loads((config.root / r["transcript_path"]).read_text())
        rows.append(
            {
                "id": r["id"],
                "split": r["split"],
                "turns": doc["turns"],
                "intent": r["labels"]["intent"],
            }
        )
    return pd.DataFrame(rows)


SPLIT_MAP = {
    "abcd": {"train": "train", "validation": "dev", "test": "test"},
    "banking77": {"train": "train", "validation": "train", "test": "test"},
}


def _bucket(key: str, seed: int) -> float:
    """Deterministic value in [0, 1) for hashing ids into train/validation."""
    return random.Random(f"{seed}:{key}").random()


def training_examples(
    config: AppConfig, split: str, datasets: list[str] | None = None
) -> pd.DataFrame:
    """(text, label, dataset) examples for ``split`` from ABCD (customer-turn prefixes) and Banking77."""
    mcfg = config.models.intent  # type: ignore[attr-defined]
    val_frac = float(config.split.ratios["validation"])  # type: ignore[attr-defined]
    seed = int(mcfg["seed"])
    out: list[dict[str, str]] = []
    for ds in datasets or list(SPLIT_MAP):
        df = load_frame(config, ds, [SPLIT_MAP[ds][split]])
        for _, r in df.iterrows():
            if ds == "banking77" and split in ("train", "validation"):
                is_val = _bucket(r["id"], seed) < val_frac
                if is_val != (split == "validation"):
                    continue
            texts = {customer_text(r["turns"], k) for k in mcfg["prefix_turns"]} | {
                customer_text(r["turns"])
            }
            out += [{"text": t, "label": r["intent"], "dataset": ds} for t in texts if t.strip()]
    return pd.DataFrame(out)
