"""M1: speaker diarization backends (pyannote, dual-channel, MFCC-clustering fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from loguru import logger

from callgraph.config import AppConfig


@dataclass
class SpeakerSegment:
    """A speaker-homogeneous time span."""

    speaker: str
    start: float
    end: float


@lru_cache(maxsize=2)
def _pyannote(name: str, token: str | None) -> Any:
    from pyannote.audio import Pipeline

    return Pipeline.from_pretrained(name, token=token)


def pyannote_available(config: AppConfig) -> bool:
    """True if a HF token is configured (the pipeline is gated)."""
    return config.hf_token is not None and bool(config.hf_token.get_secret_value())


def diarize_pyannote(
    config: AppConfig, wav: Path, params: dict[str, Any] | None = None
) -> list[SpeakerSegment]:
    """Diarize with the configured pyannote pipeline."""
    d = config.pipeline.diarization  # type: ignore[attr-defined]
    name = config.models.diarization["pipeline"]  # type: ignore[attr-defined]
    pipe = _pyannote(name, config.hf_token.get_secret_value() if config.hf_token else None)
    if params:
        pipe.instantiate(params)
    ann = pipe(str(wav), min_speakers=int(d["min_speakers"]), max_speakers=int(d["max_speakers"]))
    ann = getattr(ann, "speaker_diarization", ann)
    return [
        SpeakerSegment(spk, seg.start, seg.end) for seg, _, spk in ann.itertracks(yield_label=True)
    ]


def diarize_channels(config: AppConfig, stereo: np.ndarray, sr: int) -> list[SpeakerSegment]:
    """One speaker per channel: frame-wise louder channel (by margin) owns the frame."""
    d = config.pipeline.diarization  # type: ignore[attr-defined]
    frame = int(float(d["channel_frame_s"]) * sr)
    n = stereo.shape[1] // frame
    rms = np.stack(
        [
            np.sqrt(np.mean(stereo[c, : n * frame].reshape(n, frame) ** 2, axis=1) + 1e-10)
            for c in range(stereo.shape[0])
        ]
    )
    db = 20 * np.log10(rms + 1e-10)
    floor = float(config.pipeline.audio["silence_threshold_db"])  # type: ignore[attr-defined]
    owner = np.where(db.max(axis=0) > floor, db.argmax(axis=0), -1)
    margin = float(d["channel_energy_margin_db"])
    owner[(np.sort(db, axis=0)[-1] - np.sort(db, axis=0)[-2]) < margin] = -1
    segs: list[SpeakerSegment] = []
    i = 0
    while i < n:
        j = i
        while j < n and owner[j] == owner[i]:
            j += 1
        if owner[i] >= 0:
            segs.append(SpeakerSegment(f"SPEAKER_{owner[i]:02d}", i * frame / sr, j * frame / sr))
        i = j
    return segs


def diarize_mfcc(
    config: AppConfig, y: np.ndarray, sr: int, vad: list[tuple[float, float]]
) -> list[SpeakerSegment]:
    """Ungated fallback: cluster MFCC statistics of speech windows into ``max_speakers``."""
    from sklearn.cluster import AgglomerativeClustering

    d = config.pipeline.diarization  # type: ignore[attr-defined]
    win = float(d["mfcc_window_s"])
    pieces: list[tuple[float, float]] = []
    for s, e in vad:
        t = s
        while t < e:
            pieces.append((t, min(t + win, e)))
            t += win
    feats, keep = [], []
    for s, e in pieces:
        seg = y[int(s * sr) : int(e * sr)]
        if len(seg) < float(d["min_segment_s"]) * sr:
            continue
        m = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=int(d["mfcc_n"]))
        feats.append(np.concatenate([m.mean(axis=1), m.std(axis=1)]))
        keep.append((s, e))
    if len(feats) <= 1:
        return [SpeakerSegment("SPEAKER_00", s, e) for s, e in keep]
    x = np.array(feats)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
    labels = AgglomerativeClustering(n_clusters=int(d["max_speakers"])).fit_predict(x)
    return [
        SpeakerSegment(f"SPEAKER_{int(lab):02d}", s, e)
        for (s, e), lab in zip(keep, labels, strict=False)
    ]


def merge_segments(segs: list[SpeakerSegment], gap_s: float) -> list[SpeakerSegment]:
    """Merge adjacent same-speaker segments separated by < ``gap_s``."""
    out: list[SpeakerSegment] = []
    for s in sorted(segs, key=lambda x: x.start):
        if out and out[-1].speaker == s.speaker and s.start - out[-1].end < gap_s:
            out[-1].end = max(out[-1].end, s.end)
        else:
            out.append(SpeakerSegment(s.speaker, s.start, s.end))
    return out


def resolve_backend(config: AppConfig, stereo: bool) -> str:
    """Pick a backend: explicit config wins; 'auto' prefers channels (if enabled), pyannote, then MFCC."""
    backend = str(config.pipeline.diarization["backend"])  # type: ignore[attr-defined]
    if backend != "auto":
        return backend
    if stereo and config.features.dual_channel:  # type: ignore[attr-defined]
        return "channel"  # caller verifies channel independence (see ``channels_independent``)
    if pyannote_available(config):
        return "pyannote"
    logger.warning("No HF token: falling back to MFCC clustering diarization (not pyannote)")
    return "mfcc"


def channels_independent(config: AppConfig, stereo: np.ndarray) -> bool:
    """True if the two channels are uncorrelated enough to be one speaker each."""
    corr = abs(float(np.corrcoef(stereo[0], stereo[1])[0, 1]))
    return corr < float(config.pipeline.diarization["channel_max_corr"])  # type: ignore[attr-defined]
