"""Name normalisation: merge variants ("refund" ~ "money back") by embedding cosine similarity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from callgraph.config import AppConfig
from callgraph.utils.embeddings import embed


class Normalizer:
    """Persistent canonical-name registry per kind (issue / action / product)."""

    def __init__(self, config: AppConfig, kind: str) -> None:
        self.config = config
        self.path: Path = config.path("cache_dir") / "canonical" / f"{kind}.json"
        self.names: list[str] = json.loads(self.path.read_text()) if self.path.exists() else []
        self.vecs = embed(config, self.names) if self.names else np.zeros((0, 0))
        self.threshold = float(config.thresholds.normalization["similarity_threshold"])  # type: ignore[attr-defined]

    def canonical(self, name: str) -> str:
        """Return the existing canonical name above the threshold, else register ``name``."""
        key = " ".join(name.lower().split())
        if not key:
            return key
        v = embed(self.config, [key])
        if len(self.names):
            sims = self.vecs @ v[0]
            i = int(sims.argmax())
            if sims[i] >= self.threshold:
                return self.names[i]
        self.names.append(key)
        self.vecs = v if self.vecs.size == 0 else np.vstack([self.vecs, v])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.names))
        return key
