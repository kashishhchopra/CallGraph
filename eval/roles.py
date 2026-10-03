"""Role-assignment evaluation on SpokenWOZ (gold roles from the system/user tags): heuristic vs heuristic+LLM."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from callgraph.config import AppConfig
from callgraph.core import diarization as dz
from callgraph.core.alignment import build_turns
from callgraph.core.asr import transcribe
from callgraph.core.roles import assign_roles, first_speaker_roles
from callgraph.llm.client import LLMClient
from callgraph.pipeline import result_path
from callgraph.schemas import Turn
from eval.asr import audio_for
from eval.common import load_doc, save_result, subset


def gold_role_at(doc: dict[str, Any], t: float) -> str | None:
    for turn in doc["turns"]:
        if turn["start"] <= t <= turn["end"]:
            return turn["role"]
    return None


def time_weighted_accuracy(turns, doc: dict[str, Any], roles_by_turn: list[str]) -> float:
    ok = tot = 0.0
    for t, role in zip(turns, roles_by_turn, strict=False):
        g = gold_role_at(doc, (t.start + t.end) / 2)
        if g is None:
            continue
        w = max(t.end - t.start, 1e-6)
        tot += w
        ok += w * (g == role)
    return ok / tot if tot else float("nan")


def run(config: AppConfig) -> dict[str, Any]:
    n = int(config.evaluation.roles["subset"])  # type: ignore[attr-defined]
    llm = LLMClient(config)
    sr = int(config.pipeline.audio["target_sample_rate"])  # type: ignore[attr-defined]
    size = config.models.asr["model_size"]  # type: ignore[attr-defined]
    rows = []
    for _, row in subset(config, "spokenwoz", n).iterrows():
        doc = load_doc(config, row)
        y = audio_for(config, row, None)
        asr = transcribe(config, y, size)
        segs = dz.diarize_mfcc(config, y, sr, [(s["start"], s["end"]) for s in asr["segments"]])
        turns = build_turns(config, asr["words"], dz.merge_segments(segs, float(config.pipeline.alignment["turn_merge_gap_s"])))  # type: ignore[attr-defined]
        if not turns:
            continue
        heur = first_speaker_roles(config, turns)
        llm_turns, _ = assign_roles(config, turns, llm)
        prod = None
        rp = result_path(config, str(row["id"]))
        if rp.exists():
            pt = json.loads(rp.read_text())["transcript"]["turns"]
            pturns = [
                Turn(
                    speaker=t["speaker"],
                    role=t["role"],
                    start=t["start"],
                    end=t["end"],
                    text=t["text"],
                )
                for t in pt
            ]
            prod = time_weighted_accuracy(pturns, doc, [t.role for t in pturns])
        rows.append(
            {
                "production_pipeline": prod,
                "call_id": row["id"],
                "first_speaker": time_weighted_accuracy(
                    turns, doc, [heur[t.speaker] for t in turns]
                ),
                "first_speaker_plus_llm": time_weighted_accuracy(
                    turns, doc, [t.role for t in llm_turns]
                ),
            }
        )
    df = pd.DataFrame(rows)
    prod_df = df.dropna(subset=["production_pipeline"])
    out = {
        "n_calls": len(df),
        "production_pipeline_accuracy": (
            float(prod_df.production_pipeline.mean()) if len(prod_df) else None
        ),
        "production_pipeline_n_calls": len(prod_df),
        "first_speaker_accuracy": float(df.first_speaker.mean()),
        "first_speaker_plus_llm_accuracy": float(df.first_speaker_plus_llm.mean()),
        "note": "first_speaker columns use the MFCC fallback; production_pipeline = stored transcripts (per-channel ASR + LLM role check); gold roles from SpokenWOZ speaker tags",
    }
    save_result(config, "role_assignment", out, df)
    return out
