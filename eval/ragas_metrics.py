"""RAGAS-style faithfulness and context precision, implemented with an LLM judge (prompts in prompts/*.j2)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from callgraph.config import AppConfig
from callgraph.llm.client import LLMClient, LLMError, render_prompt


class _Flag(BaseModel):
    supported: bool


def score(config: AppConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean faithfulness / context precision over rows (each has question, answer, contexts)."""
    llm = LLMClient(config)
    cap = int((config.rag.model_extra or {})["max_context_chars"])  # type: ignore[attr-defined]
    faith, prec = [], []
    for r in rows:
        ctxs = [c for c in r["contexts"] if c]
        if not r["answer"]:
            continue
        try:
            p = render_prompt(
                "ragas_faithfulness", config, context="\n---\n".join(ctxs)[:cap], answer=r["answer"]
            )
            faith.append(float(llm.complete_model(p, _Flag).supported))  # type: ignore[attr-defined]
            if ctxs:
                rel = [
                    float(llm.complete_model(render_prompt("ragas_context", config, question=r["question"], context=c[:cap]), _Flag).supported)  # type: ignore[attr-defined]
                    for c in ctxs
                ]
                prec.append(sum(rel) / len(rel))
        except LLMError:
            continue
    return {
        "faithfulness": sum(faith) / len(faith) if faith else None,
        "context_precision": sum(prec) / len(prec) if prec else None,
        "implementation": "LLM-judge implementation of the RAGAS definitions (local Ollama judge)",
    }
