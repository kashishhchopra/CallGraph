"""ASR evaluation: WER per dataset, Whisper size vs WER vs speed (exp 1), 16 kHz vs 8 kHz (exp 2)."""

from __future__ import annotations

import time
from typing import Any

import jiwer
import librosa
import pandas as pd

from callgraph.config import AppConfig
from callgraph.core.asr import transcribe
from callgraph.core.preprocess import normalize_for_wer, simulate_telephone
from eval.common import load_doc, save_result, subset


def reference_text(dataset: str, doc: dict[str, Any], excerpt_s: float | None = None) -> str:
    """Reference transcript for a manifest document."""
    if "reference" in doc:
        return doc["reference"]
    turns = [t for t in doc["turns"] if excerpt_s is None or t["end"] <= excerpt_s]
    return " ".join(t["text"] for t in turns)


def audio_for(config: AppConfig, row: pd.Series, excerpt_s: float | None, phone: bool = False):
    sr = int(config.pipeline.audio["target_sample_rate"])  # type: ignore[attr-defined]
    y, _ = librosa.load(str(config.root / row["audio_path"]), sr=sr, mono=True, duration=excerpt_s)
    if phone:
        y = simulate_telephone(y, sr, int(config.pipeline.audio["telephone_sample_rate"]))  # type: ignore[attr-defined]
    return y


def wer_for(
    config: AppConfig, dataset: str, n: int, size: str, phone: bool = False
) -> dict[str, Any]:
    """Mean corpus WER and real-time factor for one (dataset, whisper size)."""
    ev = config.evaluation  # type: ignore[attr-defined]
    excerpt = ev.excerpt_s.get(dataset) if hasattr(ev.excerpt_s, "get") else None
    refs, hyps, audio_s, wall = [], [], 0.0, 0.0
    for _, row in subset(config, dataset, n).iterrows():
        y = audio_for(config, row, excerpt, phone)
        t0 = time.time()
        out = transcribe(config, y, size)
        wall += time.time() - t0
        audio_s += len(y) / int(config.pipeline.audio["target_sample_rate"])  # type: ignore[attr-defined]
        refs.append(normalize_for_wer(reference_text(dataset, load_doc(config, row), excerpt)))
        hyps.append(normalize_for_wer(" ".join(w.text for w in out["words"])))
    keep = [i for i, r in enumerate(refs) if r]
    return {
        "dataset": dataset,
        "size": size,
        "phone_8k": phone,
        "n": len(keep),
        "wer": jiwer.wer([refs[i] for i in keep], [hyps[i] for i in keep]),
        "rtf": wall / max(audio_s, 1e-9),
        "audio_s": audio_s,
    }


def run(config: AppConfig) -> dict[str, Any]:
    """Run ASR WER (all sizes x datasets), and the 8 kHz experiment when enabled."""
    ev = config.evaluation  # type: ignore[attr-defined]
    rows = []
    for ds, n in ev.subsets["asr"].items():
        for size in ev.asr["sizes"]:
            rows.append(wer_for(config, ds, int(n), size))
    exp8 = []
    if ev.experiments["sample_rate_16k_vs_8k"]:
        ds = ev.asr["telephone_experiment_dataset"]
        for phone in (False, True):
            exp8.append(wer_for(config, ds, int(ev.subsets["asr"][ds]), config.models.asr["model_size"], phone))  # type: ignore[attr-defined]
    table = pd.DataFrame(rows)
    save_result(config, "asr_wer", {"rows": rows}, table)
    save_result(config, "exp1_whisper_size_vs_wer_speed", {"rows": rows}, table)
    if exp8:
        save_result(config, "exp2_sample_rate_wer", {"rows": exp8}, pd.DataFrame(exp8))
    return {"asr": rows, "exp2": exp8}
