"""Diarization evaluation: DER (pyannote.metrics) on AMI, VoxConverse, SpokenWOZ, CallHome."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

from callgraph.config import AppConfig
from callgraph.core import diarization as dz
from callgraph.core.asr import transcribe
from callgraph.data.loaders.diarization import parse_rttm
from eval.asr import audio_for
from eval.common import load_doc, save_result, subset


def to_annotation(segs: list[dict[str, Any]], excerpt_s: float | None) -> Annotation:
    ann = Annotation()
    for s in segs:
        end = min(s["end"], excerpt_s) if excerpt_s else s["end"]
        if end > s["start"]:
            ann[Segment(s["start"], end)] = s["speaker"]
    return ann


def reference_segments(config: AppConfig, dataset: str, row: pd.Series) -> list[dict[str, Any]]:
    """Ground-truth speaker segments for a manifest row."""
    fmt = dict(config.rttm.model_extra or {})  # type: ignore[attr-defined]
    if dataset in ("ami", "voxconverse"):
        return parse_rttm((config.root / row["labels"]["rttm"]).read_text(), fmt)
    doc = load_doc(config, row)
    return [{"speaker": t["speaker"], "start": t["start"], "end": t["end"]} for t in doc["turns"]]


def hypothesis(
    config: AppConfig, y, row: pd.Series, backend: str, size: str, params: dict | None = None
) -> list[dict[str, Any]]:
    """Run a diarization backend on mono audio ``y`` (16 kHz)."""
    sr = int(config.pipeline.audio["target_sample_rate"])  # type: ignore[attr-defined]
    if backend == "mfcc":
        asr = transcribe(config, y, size)
        segs = dz.diarize_mfcc(config, y, sr, [(s["start"], s["end"]) for s in asr["segments"]])
    elif backend == "pyannote":
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.wav"
            sf.write(str(p), y, sr)
            segs = dz.diarize_pyannote(config, p, params)
    else:
        raise ValueError(backend)
    return [{"speaker": s.speaker, "start": s.start, "end": s.end} for s in dz.merge_segments(segs, float(config.pipeline.alignment["turn_merge_gap_s"]))]  # type: ignore[attr-defined]


def run(config: AppConfig) -> dict[str, Any]:
    """DER per (dataset, backend)."""
    ev = config.evaluation  # type: ignore[attr-defined]
    rows: list[dict[str, Any]] = []
    tuned = bool(config.models.diarization["tuned_params"])  # type: ignore[attr-defined]
    backends = ["mfcc"] + (
        ["pyannote"] + (["pyannote_tuned"] if tuned else [])
        if dz.pyannote_available(config)
        else []
    )
    for ds, n in ev.subsets["diarization"].items():
        man = (
            subset(config, ds, int(n))
            if (config.path("manifests_dir") / f"{ds}.csv").exists()
            else pd.DataFrame()
        )
        if man.empty:
            rows.append(
                {
                    "dataset": ds,
                    "backend": "-",
                    "der": None,
                    "note": "no data (see README for manual steps)",
                }
            )
            continue
        excerpt = ev.excerpt_s.get(ds) if hasattr(ev.excerpt_s, "get") else None
        for backend in backends:
            metric = DiarizationErrorRate(
                collar=float(ev.der["collar_s"]), skip_overlap=bool(ev.der["skip_overlap"])
            )
            for _, row in man.iterrows():
                y = audio_for(config, row, excerpt)
                params = dict(config.models.diarization["tuned_params"]) if backend == "pyannote_tuned" else None  # type: ignore[attr-defined]
                hyp = hypothesis(config, y, row, backend.replace("_tuned", ""), config.models.asr["model_size"], params)  # type: ignore[attr-defined]
                ref = to_annotation(reference_segments(config, ds, row), excerpt)
                metric(ref, to_annotation(hyp, excerpt), uem=Timeline([Segment(0, len(y) / int(config.pipeline.audio["target_sample_rate"]))]))  # type: ignore[attr-defined]
            rows.append(
                {"dataset": ds, "backend": backend, "n": len(man), "der": abs(metric), "note": ""}
            )
    if not tuned:
        rows.append(
            {
                "dataset": "*",
                "backend": "pyannote_tuned",
                "der": None,
                "note": "not run: no tuned parameters defined in configs/models.yaml",
            }
        )
    if not dz.pyannote_available(config):
        rows.append(
            {
                "dataset": "*",
                "backend": "pyannote",
                "der": None,
                "note": "not run: HF token / gated model terms missing",
            }
        )
    table = pd.DataFrame(rows)
    save_result(config, "diarization_der", {"rows": rows}, table)
    return {"rows": rows}
