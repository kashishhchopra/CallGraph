"""Frozen wav2vec2 embeddings (mean-pooled hidden layer) for speech emotion."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from callgraph.config import AppConfig
from callgraph.utils.device import pick_device


@lru_cache(maxsize=1)
def _load(backbone: str, dev: str):
    fe = AutoFeatureExtractor.from_pretrained(backbone)
    return fe, AutoModel.from_pretrained(backbone).to(dev).eval()


class AudioEmbedder:
    """Turns waveforms (16 kHz) into fixed-size vectors (mean + std of one hidden layer)."""

    def __init__(self, config: AppConfig) -> None:
        m = config.models.emotion_audio  # type: ignore[attr-defined]
        self.dev = str(pick_device(str(m["device"])))
        self.fe, self.model = _load(str(m["backbone"]), self.dev)
        self.layer = int(m["layer"])
        self.sr = int(self.fe.sampling_rate)

    @torch.no_grad()
    def embed(self, waves: list[np.ndarray]) -> np.ndarray:
        enc = self.fe(waves, sampling_rate=self.sr, return_tensors="pt", padding=True)
        hs = self.model(enc["input_values"].to(self.dev), output_hidden_states=True).hidden_states[
            self.layer
        ]
        return torch.cat([hs.mean(1), hs.std(1)], dim=-1).cpu().numpy()
