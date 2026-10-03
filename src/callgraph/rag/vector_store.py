"""Transcript-chunk vector store (FAISS) — idempotent by chunk id."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from callgraph.config import AppConfig
from callgraph.schemas import CallTranscript
from callgraph.utils.embeddings import embed


def chunk_call(config: AppConfig, call: CallTranscript) -> list[dict[str, Any]]:
    """Sliding windows of turns (size/overlap from config)."""
    v = config.vector_store
    size, overlap = int(v.chunk_size_turns), int(v.chunk_overlap_turns)  # type: ignore[attr-defined]
    step = max(size - overlap, 1)
    out = []
    for i in range(0, max(len(call.turns), 1), step):
        turns = call.turns[i : i + size]
        if not turns:
            break
        out.append(
            {
                "chunk_id": f"{call.call_id}#{i}",
                "call_id": call.call_id,
                "start": turns[0].start,
                "text": "\n".join(f"{t.role.upper()}: {t.text}" for t in turns),
            }
        )
        if i + size >= len(call.turns):
            break
    return out


class VectorStore:
    """FAISS inner-product index over normalised embeddings, persisted next to its metadata."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.dir: Path = config.path("processed_dir") / "vector"
        self.meta_path = self.dir / "chunks.json"
        self.chunks: dict[str, dict[str, Any]] = (
            json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {}
        )
        self._index: faiss.Index | None = None
        self._ids: list[str] = []

    def add_call(self, call: CallTranscript) -> int:
        """Add (or replace) all chunks of a call; returns the number of chunks."""
        for k in [k for k, c in self.chunks.items() if c["call_id"] == call.call_id]:
            del self.chunks[k]
        new = chunk_call(self.config, call)
        for c in new:
            self.chunks[c["chunk_id"]] = c
        self._index = None
        self.save()
        return len(new)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.chunks))

    def _signature(self) -> str:
        """Fingerprint of the stored chunks (ids and text), used to validate the on-disk embedding cache."""
        h = hashlib.sha256()
        for cid in sorted(self.chunks):
            h.update(cid.encode())
            h.update(self.chunks[cid]["text"].encode())
        return h.hexdigest()

    def _build(self) -> None:
        self._ids = list(self.chunks)
        sig = self._signature()
        cache, sigfile = self.dir / "embeddings.npy", self.dir / "embeddings.sig"
        order = self.dir / "embeddings.ids.json"
        vecs = None
        if cache.exists() and sigfile.exists() and sigfile.read_text() == sig and order.exists():
            self._ids = json.loads(order.read_text())
            vecs = np.load(cache)
        if vecs is None:
            vecs = embed(self.config, [self.chunks[i]["text"] for i in self._ids]).astype("float32")
            self.dir.mkdir(parents=True, exist_ok=True)
            np.save(cache, vecs)
            order.write_text(json.dumps(self._ids))
            sigfile.write_text(sig)
        self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)

    def search(self, query: str, k: int, call_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Top-k chunks by cosine similarity (optionally restricted to calls)."""
        if not self.chunks:
            return []
        if self._index is None:
            self._build()
        q = embed(self.config, [query]).astype("float32")
        scores, idx = self._index.search(q, min(len(self._ids), k * (10 if call_ids else 1)))  # type: ignore[union-attr]
        out = []
        for s, i in zip(scores[0], idx[0], strict=False):
            c = self.chunks[self._ids[int(i)]]
            if call_ids and c["call_id"] not in call_ids:
                continue
            out.append({**c, "score": float(s)})
            if len(out) >= k:
                break
        return out
