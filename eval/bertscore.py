"""BERTScore (Zhang et al., 2020) re-implementation: greedy cosine matching of contextual token embeddings.

The ``bert_score`` package is incompatible with the installed transformers version, so this reproduces the
metric without idf weighting and without baseline rescaling (values are therefore not comparable to rescaled scores).
"""

from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModel, AutoTokenizer

from callgraph.config import AppConfig


@lru_cache(maxsize=1)
def _load(name: str):
    return AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()


@torch.no_grad()
def _embed(config: AppConfig, texts: list[str]) -> list[torch.Tensor]:
    m = config.models.summarization  # type: ignore[attr-defined]
    tok, model = _load(str(m["bertscore_model"]))
    out = []
    for t in texts:
        enc = tok(
            t or " ",
            truncation=True,
            max_length=int(m["bertscore_max_length"]),
            return_tensors="pt",
        )
        hs = model(**enc, output_hidden_states=True).hidden_states[int(m["bertscore_layer"])][0]
        hs = torch.nn.functional.normalize(
            hs[1:-1] if len(hs) > len(enc["input_ids"][0][:2]) else hs, dim=-1
        )
        out.append(hs)
    return out


def bertscore_f1(config: AppConfig, hyps: list[str], refs: list[str]) -> list[float]:
    """Per-pair BERTScore F1."""
    scores = []
    for h, r in zip(_embed(config, hyps), _embed(config, refs), strict=True):
        sim = h @ r.T
        p, rec = sim.max(dim=1).values.mean().item(), sim.max(dim=0).values.mean().item()
        scores.append(2 * p * rec / max(p + rec, 1e-9))
    return scores
