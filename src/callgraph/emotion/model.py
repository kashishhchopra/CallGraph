"""M3 inference: per-turn emotion (audio + text late fusion) and call-level trajectory."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from callgraph.config import AppConfig
from callgraph.emotion.train import out_dir
from callgraph.registry import MODULES, AnalysisModule
from callgraph.schemas import CallTranscript, EmotionOutput, TurnEmotion
from callgraph.utils.device import pick_device


@lru_cache(maxsize=2)
def _text(path: str, dev: str) -> tuple[Any, Any, list[str]]:
    tok = AutoTokenizer.from_pretrained(path)
    return (
        tok,
        AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval(),
        json.loads(Path(path, "labels.json").read_text()),
    )


@lru_cache(maxsize=2)
def _audio_head(path: str) -> tuple[Any, list[str]]:
    import pickle

    return pickle.loads(Path(path, "head.pkl").read_bytes()), json.loads(
        Path(path, "labels.json").read_text()
    )


def load_fusion(config: AppConfig) -> dict[str, float]:
    """Fusion weights: tuned artifact (preferred) or values in thresholds.yaml if set."""
    f = config.thresholds.fusion  # type: ignore[attr-defined]
    art = config.root / f["artifact"]
    if art.exists():
        return json.loads(art.read_text())
    if f.get("audio_weight") is not None:
        return {"audio_weight": float(f["audio_weight"]), "text_weight": float(f["text_weight"])}
    raise FileNotFoundError(f"Fusion weights missing: run emotion tuning to create {art}")


class EmotionEngine:
    """Holds both branches and fuses their probabilities."""

    def __init__(self, config: AppConfig, need_audio: bool = True) -> None:
        self.config = config
        self.dev = str(pick_device(str(config.models.emotion_text["device"])))  # type: ignore[attr-defined]
        self.ttok, self.tmodel, self.labels = _text(str(out_dir(config, "emotion_text")), self.dev)
        self.fusion_tuned = True
        self.audio_ready = need_audio and (out_dir(config, "emotion_audio") / "head.pkl").exists()
        if self.audio_ready:
            from callgraph.emotion.audio_features import AudioEmbedder

            self.embedder = AudioEmbedder(config)
            self.ahead, _ = _audio_head(str(out_dir(config, "emotion_audio")))
        self.fusion = {"audio_weight": 0.0, "text_weight": 1.0}
        if self.audio_ready:
            try:
                self.fusion = load_fusion(config)
            except (
                FileNotFoundError
            ):  # first run: tuning (which needs this engine) has not produced weights yet
                self.fusion = dict(config.thresholds.fusion["untuned_default"])  # type: ignore[attr-defined]
                self.fusion_tuned = False

    @torch.no_grad()
    def text_proba(self, texts: list[str]) -> np.ndarray:
        enc = self.ttok(
            texts,
            truncation=True,
            max_length=int(self.config.models.emotion_text["max_length"]),  # type: ignore[attr-defined]
            padding=True,
            return_tensors="pt",
        ).to(self.dev)
        return torch.softmax(self.tmodel(**enc).logits, -1).cpu().numpy()

    @torch.no_grad()
    def audio_proba(self, waves: list[np.ndarray]) -> np.ndarray:
        probs = self.ahead.predict_proba(self.embedder.embed(waves))
        full = np.zeros((len(waves), len(self.labels)))
        for j, lab in enumerate(self.ahead.classes_):
            full[:, self.labels.index(lab)] = probs[:, j]
        return full

    def fuse(self, p_audio: np.ndarray | None, p_text: np.ndarray) -> np.ndarray:
        if p_audio is None:
            return p_text
        wa, wt = self.fusion["audio_weight"], self.fusion["text_weight"]
        return (wa * p_audio + wt * p_text) / (wa + wt)


def frustration_score(
    config: AppConfig, turns: list[TurnEmotion], texts: list[str]
) -> tuple[float, float]:
    """Return (frustration 0-100, peak anger) from anger prob, negative-duration ratio, escalation words."""
    e = config.thresholds.emotion  # type: ignore[attr-defined]
    w = e["frustration_weights"]
    neg = set(e["negative_labels"])
    peak = max((t.probs.get(e["anger_label"], 0.0) for t in turns), default=0.0)
    ratio = sum(1 for t in turns if t.emotion in neg) / max(len(turns), 1)
    blob = " ".join(texts).lower()
    hits = sum(1 for k in e["escalation_words"] if k in blob)
    esc = min(hits / float(e["escalation_words_cap"]), 1.0)
    return (
        float(
            w["peak_anger"] * peak
            + w["negative_duration_ratio"] * ratio
            + w["escalation_words"] * esc
        ),
        peak,
    )


def summarize_trajectory(
    config: AppConfig, call_id: str, turns: list[TurnEmotion], texts: list[str]
) -> EmotionOutput:
    """Call-level features from per-turn emotions."""
    e = config.thresholds.emotion  # type: ignore[attr-defined]
    if not turns:
        return EmotionOutput(call_id=call_id, turns=[])
    pos, neg = set(e["positive_labels"]), set(e["negative_labels"])
    valence = lambda t: sum(p for k, p in t.probs.items() if k in pos) - sum(  # noqa: E731
        p for k, p in t.probs.items() if k in neg
    )
    delta = valence(turns[-1]) - valence(turns[0])
    margin = float(e["shift_margin"])
    shift = "improved" if delta > margin else "worsened" if delta < -margin else "stable"
    score, peak = frustration_score(config, turns, texts)
    first, last = turns[0], turns[-1]
    return EmotionOutput(
        call_id=call_id,
        turns=turns,
        start_emotion=first.emotion,
        end_emotion=last.emotion,
        start_intensity=first.intensity,
        end_intensity=last.intensity,
        peak_anger=peak,
        shift=shift,
        frustration_score=score,
        ends_negative=last.emotion in neg,
    )


@MODULES.register("emotion")
class EmotionModule(AnalysisModule):
    """Registered M3 plugin. ``context['audio_path']`` enables the audio branch."""

    _engine: EmotionEngine | None = None

    def run(self, call: CallTranscript, config: AppConfig, **context: Any) -> EmotionOutput:
        if self._engine is None:
            type(self)._engine = EmotionEngine(config)
        eng = self._engine
        idx = [i for i, t in enumerate(call.turns) if t.role == "customer" and t.text.strip()]
        texts = [call.turns[i].text for i in idx]
        if not idx:
            return EmotionOutput(call_id=call.call_id, turns=[])
        p_text = eng.text_proba(texts)  # type: ignore[union-attr]
        p_audio = None
        audio = context.get("audio_path")
        if eng.audio_ready and audio:  # type: ignore[union-attr]
            sr = int(eng.embedder.sr)  # type: ignore[union-attr]
            y, _ = librosa.load(str(audio), sr=sr, mono=True)
            max_n = int(float(config.models.emotion_audio["max_audio_s"]) * sr)  # type: ignore[attr-defined]
            waves = [
                y[int(call.turns[i].start * sr) : int(call.turns[i].end * sr)][:max_n] for i in idx
            ]
            waves = [w if len(w) else np.zeros(sr) for w in waves]
            p_audio = eng.audio_proba(waves)  # type: ignore[union-attr]
        probs = eng.fuse(p_audio, p_text)  # type: ignore[union-attr]
        turns = [
            TurnEmotion(
                turn_index=i,
                emotion=eng.labels[int(p.argmax())],
                intensity=float(p.max()),  # type: ignore[union-attr]
                probs={lab: float(v) for lab, v in zip(eng.labels, p, strict=False)},
            )  # type: ignore[union-attr]
            for i, p in zip(idx, probs, strict=False)
        ]
        return summarize_trajectory(config, call.call_id, turns, texts)
