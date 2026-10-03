"""Late-fusion weight tuning on the MELD validation split (audio + text)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from callgraph.config import AppConfig
from callgraph.data.manifest import read_manifest
from callgraph.emotion.model import EmotionEngine
from callgraph.emotion.train import load_audio_clip


def split_probs(
    config: AppConfig,
    eng: EmotionEngine,
    rows: pd.DataFrame,
    phone: bool = False,
    use_text: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(audio_probs, text_probs) for manifest rows (text only if the manifest carries it)."""
    bs = int(config.evaluation.emotion["batch_size"])  # type: ignore[attr-defined]
    max_s = float(config.models.emotion_audio["max_audio_s"])  # type: ignore[attr-defined]
    pa, pt = [], []
    for i in range(0, len(rows), bs):
        chunk = rows.iloc[i : i + bs]
        if eng.audio_ready:
            pa.append(
                eng.audio_proba(
                    [
                        load_audio_clip(config, r.audio_path, max_s, phone)
                        for r in chunk.itertuples()
                    ]
                )
            )
        if use_text:
            pt.append(eng.text_proba([lab["text"] for lab in chunk.labels]))
    return (np.concatenate(pa) if pa else None), (np.concatenate(pt) if pt else None)


def tune_fusion(config: AppConfig) -> dict[str, Any]:
    """Grid-search the audio weight on MELD validation; write the artifact configured in thresholds.yaml."""
    eng = EmotionEngine(config)
    val = read_manifest(config, "meld")
    val = val[val.split == "validation"].head(int(config.evaluation.emotion["max_clips"]["meld"]))  # type: ignore[attr-defined]
    pa, pt = split_probs(config, eng, val)
    y = [eng.labels.index(lab["emotion"]) for lab in val.labels]
    step = float(config.thresholds.fusion["weight_grid_step"])  # type: ignore[attr-defined]
    best = {"audio_weight": 0.0, "text_weight": 1.0, "weighted_f1": -1.0}
    grid = []
    for wa in np.arange(0.0, 1.0 + step / 2, step):
        p = wa * pa + (1 - wa) * pt
        f1 = f1_score(y, p.argmax(1), average="weighted")
        grid.append({"audio_weight": float(wa), "weighted_f1": float(f1)})
        if f1 > best["weighted_f1"]:
            best = {
                "audio_weight": float(wa),
                "text_weight": float(1 - wa),
                "weighted_f1": float(f1),
            }
    art = config.root / config.thresholds.fusion["artifact"]  # type: ignore[attr-defined]
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps({**best, "n_val": len(val), "grid": grid}))
    return best
