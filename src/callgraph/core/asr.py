"""M1: faster-whisper ASR with word timestamps, language detection and confidences."""

from __future__ import annotations

import hashlib
import json
import math
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from callgraph.config import AppConfig
from callgraph.schemas import Word


@cache
def _model(size: str, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel

    logger.info("Loading faster-whisper '{}' ({}, {})", size, device, compute_type)
    return WhisperModel(size, device=device, compute_type=compute_type)


def transcribe(
    config: AppConfig, audio: np.ndarray | str | Path, model_size: str | None = None
) -> dict[str, Any]:
    """Transcribe mono 16 kHz audio. Returns words, segments, language, language_prob."""
    cfg = config.pipeline.asr  # type: ignore[attr-defined]
    size = model_size or config.models.asr["model_size"]  # type: ignore[attr-defined]
    cache: Path | None = None
    if isinstance(audio, np.ndarray):
        key = hashlib.sha256(
            audio.astype(np.float32).tobytes()
            + f"{size}{cfg['beam_size']}{cfg['vad_filter']}".encode()
        ).hexdigest()
        cache = config.path("cache_dir") / "asr" / f"{key}.json"
        if cache.exists():
            d = json.loads(cache.read_text())
            d["words"] = [Word(**w) for w in d["words"]]
            return d
    model = _model(size, cfg["device"], cfg["compute_type"])
    src = str(audio) if isinstance(audio, (str, Path)) else audio.astype(np.float32)
    segs, info = model.transcribe(
        src,
        beam_size=int(cfg["beam_size"]),
        vad_filter=bool(cfg["vad_filter"]),
        word_timestamps=bool(cfg["word_timestamps"]),
        language=None,
    )
    words: list[Word] = []
    segments: list[dict[str, Any]] = []
    for seg in segs:
        sw = [
            Word(
                text=w.word.strip(),
                start=float(w.start),
                end=float(w.end),
                prob=float(w.probability),
            )
            for w in (seg.words or [])
        ]
        words += sw
        segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "confidence": float(math.exp(seg.avg_logprob)),
                "no_speech_prob": float(seg.no_speech_prob),
            }
        )
    result = {
        "words": words,
        "segments": segments,
        "language": info.language,
        "language_prob": float(info.language_probability),
        "duration_s": float(info.duration),
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({**result, "words": [w.model_dump() for w in words]}))
    return result
