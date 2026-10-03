"""M1 orchestration: audio -> speaker-labelled CallTranscript (cached by audio hash)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf
from loguru import logger

from callgraph.config import AppConfig
from callgraph.core import diarization as dz
from callgraph.core.alignment import build_turns
from callgraph.core.asr import transcribe
from callgraph.core.preprocess import audio_hash, normalize_loudness
from callgraph.core.roles import assign_roles
from callgraph.llm.client import LLMClient
from callgraph.schemas import CallTranscript


def cache_path(config: AppConfig, ahash: str, tag: str) -> Path:
    """Cache file for a (audio hash, settings tag)."""
    return config.path("cache_dir") / "transcripts" / f"{ahash}_{tag}.json"


def run_core(
    config: AppConfig,
    audio: Path,
    call_id: str,
    dataset: str | None = None,
    metadata: dict[str, Any] | None = None,
    llm: LLMClient | None = None,
    asr_size: str | None = None,
    use_cache: bool = True,
) -> CallTranscript:
    """Transcribe, diarize, align and role-label one call (idempotent via cache)."""
    pcfg = config.pipeline  # type: ignore[attr-defined]
    size = asr_size or config.models.asr["model_size"]  # type: ignore[attr-defined]
    sr = int(pcfg.audio["target_sample_rate"])
    stereo_y, _ = librosa.load(str(audio), sr=sr, mono=False)
    stereo = stereo_y.ndim > 1 and stereo_y.shape[0] > 1
    backend = dz.resolve_backend(config, stereo)
    if backend == "channel" and not dz.channels_independent(config, stereo_y):
        backend = "pyannote" if dz.pyannote_available(config) else "mfcc"
    tag = f"{size}_{backend}_{int(bool(config.features.dual_channel))}"  # type: ignore[attr-defined]
    ahash = audio_hash(audio)
    cp = cache_path(config, ahash, tag)
    if use_cache and cp.exists():
        call = CallTranscript.model_validate_json(cp.read_text())
        # roles depend on config (window size) and the LLM check, not on audio: always recompute them
        relabelled, _ = assign_roles(config, call.turns, llm)
        call = call.model_copy(update={"turns": relabelled})
        return call.model_copy(
            update={
                "call_id": call_id,
                "dataset": dataset,
                "metadata": {**call.metadata, **(metadata or {})},
            }
        )
    mono = stereo_y.mean(axis=0) if stereo else stereo_y
    if pcfg.audio["normalize_loudness"]:
        mono = normalize_loudness(mono, float(pcfg.audio["target_dbfs"]))
    if backend == "channel":
        words, asr = [], None
        segs = []
        for ch in range(stereo_y.shape[0]):
            y = (
                normalize_loudness(stereo_y[ch], float(pcfg.audio["target_dbfs"]))
                if pcfg.audio["normalize_loudness"]
                else stereo_y[ch]
            )
            res = transcribe(config, y, size)
            asr = asr or res
            for w in res["words"]:
                words.append(w.model_copy(update={"speaker": f"CH{ch}"}))
            segs += [dz.SpeakerSegment(f"CH{ch}", w.start, w.end) for w in res["words"]]
        asr = {**asr, "words": words, "duration_s": stereo_y.shape[1] / sr}
    else:
        asr = transcribe(config, mono, size)
        vad = [(s["start"], s["end"]) for s in asr["segments"]]
        segs = None
        if backend == "pyannote":
            try:
                with tempfile.TemporaryDirectory() as td:
                    tmp = Path(td) / "a.wav"
                    sf.write(str(tmp), mono, sr)
                    segs = dz.diarize_pyannote(config, tmp)
            except Exception as exc:  # gated model terms not accepted, offline, ...
                logger.warning(
                    "pyannote failed ({}); falling back to MFCC diarization",
                    str(exc)[: int(pcfg.diarization["error_trunc_chars"])],
                )
                backend = "mfcc"
        if segs is None:
            segs = dz.diarize_mfcc(config, mono, sr, vad)
    cp = cache_path(config, ahash, f"{size}_{backend}_{int(bool(config.features.dual_channel))}")  # type: ignore[attr-defined]
    turns = build_turns(
        config, asr["words"], dz.merge_segments(segs, float(pcfg.alignment["turn_merge_gap_s"]))
    )
    turns, _ = assign_roles(config, turns, llm)
    call = CallTranscript(
        call_id=call_id,
        duration_s=asr["duration_s"],
        language=asr["language"],
        language_prob=asr["language_prob"],
        non_english=asr["language"] != pcfg.asr["expected_language"]
        or asr["language_prob"] < float(pcfg.asr["language_detection_min_prob"]),
        dataset=dataset,
        metadata={**(metadata or {}), "diarization_backend": backend, "asr_model": size},
        turns=turns,
    )
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(call.model_dump_json())
    logger.info("M1 {}: {} turns, backend={}", call_id, len(turns), backend)
    return call


def save_transcript(config: AppConfig, call: CallTranscript) -> Path:
    """Write ``data/processed/calls/<id>/transcript.json``."""
    p = config.path("processed_dir") / "calls" / call.call_id / "transcript.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(call.model_dump(), ensure_ascii=False))
    return p
