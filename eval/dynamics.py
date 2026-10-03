"""Dynamics evaluation: overlap/interruption precision-recall vs AMI ground truth (+ energy baseline)."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np
import pandas as pd

from callgraph.config import AppConfig
from callgraph.core import diarization as dz
from callgraph.core.alignment import build_turns
from callgraph.core.asr import transcribe
from callgraph.dynamics.features import overlaps
from callgraph.schemas import Turn
from eval.asr import audio_for
from eval.common import load_doc, save_result, subset
from eval.diarization import hypothesis, reference_segments

Interval = tuple[float, float]


def overlap_intervals(segs: list[dict[str, Any]], excerpt: float | None) -> list[Interval]:
    """Intervals where >= 2 distinct speakers are simultaneously active."""
    ev = sorted(
        [(s["start"], 1, s["speaker"]) for s in segs] + [(s["end"], -1, s["speaker"]) for s in segs]
    )
    active: dict[str, int] = {}
    out: list[Interval] = []
    start: float | None = None
    for t, kind, spk in ev:
        active[spk] = active.get(spk, 0) + kind
        n = sum(1 for v in active.values() if v > 0)
        if n >= 2 and start is None:  # noqa: PLR2004
            start = t
        elif n < 2 and start is not None:  # noqa: PLR2004
            out.append((start, t))
            start = None
    return [(a, min(b, excerpt) if excerpt else b) for a, b in out if not excerpt or a < excerpt]


def total(iv: list[Interval]) -> float:
    return float(sum(b - a for a, b in iv))


def intersect(a: list[Interval], b: list[Interval]) -> float:
    return float(sum(max(0.0, min(x1, y1) - max(x0, y0)) for x0, x1 in a for y0, y1 in b))


def prf(det: list[Interval], gt: list[Interval]) -> dict[str, float]:
    inter = intersect(det, gt)
    p, r = inter / max(total(det), 1e-9), inter / max(total(gt), 1e-9)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}


def event_prf(det: list[float], gt: list[float], tol: float) -> dict[str, float]:
    used, hit = set(), 0
    for d in det:
        j = next((j for j, g in enumerate(gt) if j not in used and abs(g - d) <= tol), None)
        if j is not None:
            used.add(j)
            hit += 1
    p, r = hit / max(len(det), 1), hit / max(len(gt), 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}


def energy_overlap(config: AppConfig, y: np.ndarray, sr: int) -> list[Interval]:
    """Baseline: frames whose energy exceeds the speech median by a margin are 'overlap'."""
    d = config.evaluation.dynamics  # type: ignore[attr-defined]
    hop = int(float(d["energy_frame_s"]) * sr)
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-10)
    speech = db[db > float(d["energy_floor_db"])]
    thr = (np.median(speech) if len(speech) else 0.0) + float(d["energy_margin_db"])
    frame = float(d["energy_frame_s"])
    flags = db > thr
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i * frame
        elif not f and start is not None:
            out.append((start, i * frame))
            start = None
    return out


def interruptions(turns: list[Turn], thr: float) -> list[float]:
    return [o["start"] for o in overlaps(turns, thr) if o["interruption"]]


def run(config: AppConfig) -> dict[str, Any]:
    ev = config.evaluation  # type: ignore[attr-defined]
    thr = float(config.thresholds.dynamics["interruption_overlap_s"])  # type: ignore[attr-defined]
    tol = float(ev.dynamics["match_tolerance_s"])
    sr = int(config.pipeline.audio["target_sample_rate"])  # type: ignore[attr-defined]
    excerpt = float(ev.excerpt_s["ami"])
    agg: dict[str, dict[str, list[Any]]] = {}
    for _, row in subset(config, "ami", int(ev.subsets["diarization"]["ami"])).iterrows():
        gt_segs = reference_segments(config, "ami", row)
        gt_iv = overlap_intervals(gt_segs, excerpt)
        gt_turns = [
            Turn(speaker=s["speaker"], role="x", start=s["start"], end=s["end"], text="")
            for s in sorted(gt_segs, key=lambda s: s["start"])
        ]
        gt_int = interruptions([t for t in gt_turns if t.start < excerpt], thr)
        y = audio_for(config, row, excerpt)
        det: dict[str, tuple[list[Interval], list[float]]] = {}
        oracle = [
            Turn(speaker=t["speaker"], role="x", start=t["start"], end=t["end"], text=t["text"])
            for t in load_doc(config, row)["turns"]
            if t["start"] < excerpt
        ]
        det["rule_oracle_timings"] = (
            [(o["start"], o["end"]) for o in overlaps(oracle, 0.0)],
            interruptions(oracle, thr),
        )
        asr = transcribe(config, y, config.models.asr["model_size"])  # type: ignore[attr-defined]
        for backend in ("mfcc", "pyannote") if dz.pyannote_available(config) else ("mfcc",):
            segs = hypothesis(config, y, row, backend, config.models.asr["model_size"])  # type: ignore[attr-defined]
            turns = build_turns(
                config,
                asr["words"],
                [dz.SpeakerSegment(s["speaker"], s["start"], s["end"]) for s in segs],
            )
            seg_turns = [
                Turn(speaker=g["speaker"], role="x", start=g["start"], end=g["end"], text="")
                for g in sorted(segs, key=lambda g: g["start"])
            ]
            det[f"rule_on_diarization_segments_{backend}"] = (
                overlap_intervals(segs, excerpt),
                interruptions(seg_turns, thr),
            )
            det[f"rule_pipeline_{backend}"] = (
                [(o["start"], o["end"]) for o in overlaps(turns, 0.0)],
                interruptions(turns, thr),
            )
        det["energy_baseline"] = (
            energy_overlap(config, y, sr),
            [a for a, _ in energy_overlap(config, y, sr)],
        )
        for name, (iv, ints) in det.items():
            a = agg.setdefault(name, {"iv_det": [], "ints_det": []})
            a.setdefault("pairs", []).append((prf(iv, gt_iv), event_prf(ints, gt_int, tol)))
    rows = []
    for name, a in agg.items():
        ov = pd.DataFrame([p[0] for p in a["pairs"]]).mean()
        it = pd.DataFrame([p[1] for p in a["pairs"]]).mean()
        rows.append(
            {
                "detector": name,
                "overlap_precision": ov.precision,
                "overlap_recall": ov.recall,
                "overlap_f1": ov.f1,
                "interrupt_precision": it.precision,
                "interrupt_recall": it.recall,
                "interrupt_f1": it.f1,
            }
        )
    save_result(config, "dynamics_ami", {"rows": rows}, pd.DataFrame(rows))
    return {"rows": rows}
