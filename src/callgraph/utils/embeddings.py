"""Sentence-embedding helper (model id from config)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from callgraph.config import AppConfig


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(config: AppConfig, texts: list[str]) -> np.ndarray:
    """L2-normalised embeddings, shape (n, dim)."""
    name = config.models.embedding["model"]  # type: ignore[attr-defined]
    return np.asarray(
        _model(name).encode(texts, normalize_embeddings=True, show_progress_bar=False)
    )
