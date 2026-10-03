"""M8: Vector RAG, GraphRAG (text-to-Cypher) and Hybrid strategies behind one interface."""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from callgraph.config import AppConfig
from callgraph.graph.client import GraphClient
from callgraph.llm.client import LLMClient, LLMError, render_prompt
from callgraph.rag.vector_store import VectorStore

STRATEGIES: dict[str, type[RagStrategy]] = {}


class RagAnswer(BaseModel):
    """Uniform result for all strategies."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    cypher: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    latency_s: float = 0.0
    strategy: str = ""
    fallback_used: bool = False
    error: str | None = None


class _CypherReply(BaseModel):
    cypher: str
    params: dict[str, Any] = Field(default_factory=dict)


def register(name: str):
    def deco(cls: type[RagStrategy]) -> type[RagStrategy]:
        cls.name = name
        STRATEGIES[name] = cls
        return cls

    return deco


class RagStrategy(ABC):
    """``ask(question) -> RagAnswer``."""

    name = ""

    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient | None = None,
        graph: GraphClient | None = None,
        vectors: VectorStore | None = None,
    ) -> None:
        self.config = config
        self.llm = llm or LLMClient(config)
        self._graph, self._vectors = graph, vectors
        self.cfg: dict[str, Any] = dict(config.rag.model_extra or {})  # type: ignore[attr-defined]

    @property
    def graph(self) -> GraphClient:
        if self._graph is None:
            self._graph = GraphClient(self.config)
        return self._graph

    @property
    def vectors(self) -> VectorStore:
        if self._vectors is None:
            self._vectors = VectorStore(self.config)
        return self._vectors

    def _write_answer(
        self, question: str, rows: list[dict[str, Any]], snippets: list[dict[str, Any]]
    ) -> str:
        cap = int(self.cfg["max_snippet_chars"])
        snips = [{"call_id": s["call_id"], "text": s["text"][:cap]} for s in snippets]
        prompt = render_prompt(
            "answer",
            self.config,
            question=question,
            rows=rows[: int(self.cfg["max_rows_in_prompt"])] or None,
            snippets=snips,
        )
        return self.llm.complete(prompt).strip()

    @abstractmethod
    def _ask(self, question: str) -> RagAnswer: ...

    def ask(self, question: str) -> RagAnswer:
        t0 = time.time()
        try:
            res = self._ask(question)
        except LLMError as exc:
            res = RagAnswer(answer="", error=str(exc))
        res.latency_s = time.time() - t0
        res.strategy = self.name
        return res


@register("vector")
class VectorRag(RagStrategy):
    """Embed the question, retrieve transcript chunks, answer from them."""

    def _ask(self, question: str) -> RagAnswer:
        hits = self.vectors.search(question, int(self.cfg["top_k_chunks"]))
        return RagAnswer(
            answer=self._write_answer(question, [], hits),
            sources=sorted({h["call_id"] for h in hits}),
            contexts=[h["text"] for h in hits],
        )


class GraphMixin(RagStrategy):
    """Shared text-to-Cypher machinery."""

    def known_values(self) -> dict[str, list[str]]:
        """Distinct values of low-cardinality nodes (from the live graph) to ground the generated Cypher."""
        out: dict[str, list[str]] = {}
        lim = int(self.cfg["known_value_limit"])
        for label in ("Intent", "Emotion", "Outcome", "Issue", "Action"):
            key = self.graph.schema.nodes[label].key
            try:
                rows = self.graph.ro_query(
                    f"MATCH (n:{label}) RETURN n.{key} AS v LIMIT $lim", {"lim": lim}
                )
                out[f"{label}.{key}"] = [str(r["v"]) for r in rows]
            except Exception as exc:
                logger.debug("known values {} failed: {}", label, exc)
        rub = [i["id"] for i in self.config.rubric.items]  # type: ignore[attr-defined]
        out["ComplianceCheck.rubric_item"] = rub
        return out

    def safe(self, cypher: str) -> bool:
        cleaned = re.sub(r":\s*\w+", "", cypher)  # drop labels/rel types (e.g. the :Call label)
        words = {w.upper() for w in re.findall(r"[A-Za-z_]+", cleaned)}
        return not words & {w.upper() for w in self.cfg["forbidden_cypher"]}

    def run_graph(self, question: str) -> tuple[str | None, list[dict[str, Any]], str | None]:
        """Generate -> execute -> retry with the error message. Returns (cypher, rows, last_error)."""
        known = self.known_values()
        error, previous = None, None
        cypher: str | None = None
        for _ in range(int(self.cfg["max_cypher_retries"]) + 1):
            prompt = render_prompt(
                "text_to_cypher",
                self.config,
                schema_text=self.graph.schema.schema_text(),
                default_limit=self.cfg["default_limit"],
                known_values=known,
                examples=self.cfg["examples"],
                error=error,
                previous_cypher=previous,
                question=question,
            )
            reply = self.llm.complete_model(prompt, _CypherReply)
            cypher = reply.cypher  # type: ignore[attr-defined]
            params = {"limit": int(self.cfg["default_limit"]), **reply.params}  # type: ignore[attr-defined]
            if not self.safe(cypher):
                error, previous = "Only read-only MATCH queries are allowed.", cypher
                continue
            try:
                return cypher, self.graph.ro_query(cypher, params), None
            except Exception as exc:
                error, previous = str(exc), cypher
                logger.warning("Cypher failed: {}", exc)
        return cypher, [], error

    @staticmethod
    def call_ids(rows: list[dict[str, Any]]) -> list[str]:
        ids: list[str] = []
        for r in rows:
            for k, v in r.items():
                if "call_id" in k.lower() and isinstance(v, str):
                    ids.append(v)
        return list(dict.fromkeys(ids))


@register("graph")
class GraphRag(GraphMixin):
    """Question -> Cypher -> rows (+ linked snippets) -> cited answer; vector fallback on empty results."""

    def _ask(self, question: str) -> RagAnswer:
        cypher, rows, error = self.run_graph(question)
        ids = self.call_ids(rows)
        if not rows:
            hits = self.vectors.search(question, int(self.cfg["top_k_chunks"]))
            return RagAnswer(
                answer=self._write_answer(question, [], hits),
                sources=sorted({h["call_id"] for h in hits}),
                cypher=cypher,
                contexts=[h["text"] for h in hits],
                fallback_used=True,
                error=error,
            )
        snippets = self.vectors.search(question, int(self.cfg["top_k_chunks"]), ids) if ids else []
        return RagAnswer(
            answer=self._write_answer(question, rows, snippets),
            sources=ids,
            cypher=cypher,
            rows=rows,
            contexts=[str(rows[: int(self.cfg["max_rows_in_prompt"])])]
            + [s["text"] for s in snippets],
        )


@register("hybrid")
class HybridRag(GraphMixin):
    """Graph rows plus vector-retrieved chunks together."""

    def _ask(self, question: str) -> RagAnswer:
        cypher, rows, error = self.run_graph(question)
        ids = self.call_ids(rows)
        hits = self.vectors.search(question, int(self.cfg["top_k_chunks"]))
        linked = self.vectors.search(question, int(self.cfg["top_k_chunks"]), ids) if ids else []
        seen, snippets = set(), []
        for h in [*linked, *hits]:
            if h["chunk_id"] not in seen:
                seen.add(h["chunk_id"])
                snippets.append(h)
        return RagAnswer(
            answer=self._write_answer(question, rows, snippets),
            sources=list(dict.fromkeys(ids + [h["call_id"] for h in snippets])),
            cypher=cypher,
            rows=rows,
            contexts=([str(rows[: int(self.cfg["max_rows_in_prompt"])])] if rows else [])
            + [s["text"] for s in snippets],
            error=error,
        )


def get_strategy(config: AppConfig, name: str, **kw: Any) -> RagStrategy:
    """Instantiate a strategy by name (must be listed in ``rag.yaml``)."""
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name](config, **kw)
