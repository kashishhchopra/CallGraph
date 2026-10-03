"""Emotion evaluation: weighted F1 per dataset, ablation (audio/text/fused), 8 kHz experiment, real-call set."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.metrics import f1_score

from callgraph.config import AppConfig
from callgraph.data.manifest import read_manifest
from callgraph.emotion.model import EmotionEngine, load_fusion
from callgraph.emotion.tune import split_probs
from eval.common import save_result


def wf1(y: list[str], pred: list[str]) -> float:
    return float(f1_score(y, pred, average="weighted"))


def run(config: AppConfig) -> dict[str, Any]:
    ev = config.evaluation  # type: ignore[attr-defined]
    eng = EmotionEngine(config)
    fusion = load_fusion(config)
    lim = ev.emotion["max_clips"]
    rows: list[dict[str, Any]] = []
    for ds in ("crema_d", "ravdess", "meld"):
        man = read_manifest(config, ds)
        man = man[man.split == "test"].head(int(lim[ds]))
        y = [lab["emotion"] for lab in man.labels]
        has_text = ds == "meld"
        for phone in ((False, True) if ev.experiments["sample_rate_16k_vs_8k"] else (False,)):
            pa, pt = split_probs(config, eng, man, phone=phone, use_text=has_text)
            variants = {"audio_only": pa}
            if has_text:
                variants["text_only"] = pt
                variants["fused"] = fusion["audio_weight"] * pa + fusion["text_weight"] * pt
            for name, p in variants.items():
                if p is None or (name == "text_only" and phone):
                    continue
                rows.append(
                    {
                        "dataset": ds,
                        "variant": name,
                        "phone_8k": phone,
                        "n": len(man),
                        "weighted_f1": wf1(y, [eng.labels[i] for i in p.argmax(1)]),
                    }
                )
    table = pd.DataFrame(rows)
    save_result(config, "emotion_f1", {"rows": rows, "fusion": fusion}, table)
    save_result(
        config,
        "exp4_emotion_ablation",
        {"rows": [r for r in rows if r["dataset"] == "meld" and not r["phone_8k"]]},
        table[(table.dataset == "meld") & (~table.phone_8k)],
    )
    save_result(
        config,
        "exp2_sample_rate_emotion",
        {"rows": [r for r in rows if r["variant"] == "audio_only"]},
        table[table.variant == "audio_only"],
    )
    return {"rows": rows}


def run_real(config: AppConfig) -> dict[str, Any]:
    """F1 on the silver real-call emotion set + domain-gap experiment (exp 5)."""
    path = config.path("labels_dir") / "real_call_emotion.json"
    if not path.exists():
        return {"note": "silver real-call emotion labels not built yet"}
    labels = json.loads(path.read_text())
    return {"n": len(labels["items"]), "note": "computed by eval.real_world"}
