"""M1 preprocessing: mono, resample, loudness-normalise, trim long silences, 8 kHz simulation."""

from __future__ import annotations

import hashlib
import re
import string
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from callgraph.config import AppConfig

EPS = 1e-10


def audio_hash(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of the audio bytes (cache key)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_audio(path: Path, sr: int, mono: bool = True) -> np.ndarray:
    """Load audio resampled to ``sr``; returns (n,) mono or (channels, n)."""
    y, _ = librosa.load(str(path), sr=sr, mono=mono)
    return y


def normalize_loudness(y: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale so the RMS level equals ``target_dbfs``; clip-safe."""
    rms = float(np.sqrt(np.mean(np.square(y)) + EPS))
    gain = 10 ** (target_dbfs / 20) / max(rms, EPS)
    out = y * gain
    peak = float(np.max(np.abs(out)) + EPS)
    return out / peak if peak > 1.0 else out


def trim_long_silences(
    y: np.ndarray,
    sr: int,
    min_silence_s: float,
    threshold_db: float,
    keep_s: float,
    frame_s: float,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Shorten silences longer than ``min_silence_s`` to ``keep_s``.

    Returns the audio and the list of (original_start_s, removed_s) offsets so timestamps
    can be mapped back to the source recording.
    """
    frame = int(frame_s * sr) or 1
    n = len(y) // frame
    if n == 0:
        return y, []
    rms = np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1) + EPS)
    silent = 20 * np.log10(rms + EPS) < threshold_db
    keep: list[np.ndarray] = []
    removed: list[tuple[float, float]] = []
    i = 0
    min_frames = int(min_silence_s * sr / frame)
    keep_frames = int(keep_s * sr / frame)
    while i < n:
        j = i
        while j < n and silent[j] == silent[i]:
            j += 1
        seg = y[i * frame : j * frame]
        if silent[i] and (j - i) >= min_frames:
            keep.append(seg[: keep_frames * frame])
            removed.append((i * frame / sr + keep_s, (j - i - keep_frames) * frame / sr))
        else:
            keep.append(seg)
        i = j
    keep.append(y[n * frame :])
    return np.concatenate(keep), removed


def simulate_telephone(y: np.ndarray, sr: int, phone_sr: int) -> np.ndarray:
    """Downsample to ``phone_sr`` then upsample back to ``sr`` (band-limited copy)."""
    low = librosa.resample(y, orig_sr=sr, target_sr=phone_sr)
    return librosa.resample(low, orig_sr=phone_sr, target_sr=sr)


def preprocess_file(
    config: AppConfig, src: Path, dst: Path, trim: bool = True, phone: bool = False
) -> dict[str, float]:
    """Preprocess ``src`` into a mono target-rate WAV at ``dst``; returns stats."""
    a = config.pipeline.audio  # type: ignore[attr-defined]
    sr = int(a["target_sample_rate"])
    y = load_audio(src, sr, mono=bool(a["mono"]))
    if phone:
        y = simulate_telephone(y, sr, int(a["telephone_sample_rate"]))
    if a["normalize_loudness"]:
        y = normalize_loudness(y, float(a["target_dbfs"]))
    removed: list[tuple[float, float]] = []
    if trim:
        y, removed = trim_long_silences(
            y,
            sr,
            float(a["trim_silence_min_s"]),
            float(a["silence_threshold_db"]),
            float(a["keep_silence_s"]),
            float(a["frame_s"]),
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, sr)
    return {"duration_s": len(y) / sr, "removed_s": float(sum(r for _, r in removed))}


_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize_for_wer(text: str) -> str:
    """Lowercase and strip punctuation (WER scoring only)."""
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", text.lower())).strip()
