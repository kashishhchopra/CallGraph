"""Pipeline runner: M1 -> analysis modules -> graph + vector store, for one call or a folder."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

import callgraph.compliance.audit  # noqa: F401  (registers modules)
import callgraph.dynamics.features  # noqa: F401
import callgraph.emotion.model  # noqa: F401
import callgraph.extraction.extract  # noqa: F401
import callgraph.intent.model  # noqa: F401
import callgraph.summarization.summarize  # noqa: F401
from callgraph.config import AppConfig, get_config
from callgraph.core.m1 import run_core
from callgraph.graph.client import GraphClient
from callgraph.graph.writer import GraphWriter
from callgraph.llm.client import LLMClient
from callgraph.privacy.redact import redact_call
from callgraph.rag.vector_store import VectorStore
from callgraph.registry import MODULES
from callgraph.schemas import CallResult, CallTranscript, Turn

Progress = Callable[[str, str], None]


def call_dir(config: AppConfig, call_id: str) -> Path:
    return config.path("processed_dir") / "calls" / call_id


def result_path(config: AppConfig, call_id: str) -> Path:
    return call_dir(config, call_id) / "result.json"


def write_status(config: AppConfig, call_id: str, status: dict[str, str]) -> None:
    p = call_dir(config, call_id) / "status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(status))


def text_call(
    config: AppConfig, doc: dict[str, Any], dataset: str, metadata: dict[str, Any] | None = None
) -> CallTranscript:
    """Build a transcript from a text dialogue (no audio): turns get ordinal pseudo-times."""
    turns = [
        Turn(
            speaker=t["speaker"],
            role=t.get("role", t["speaker"]),
            start=float(i),
            end=float(i),
            text=t["text"],
        )
        for i, t in enumerate(doc["turns"])
    ]
    return CallTranscript(
        call_id=doc["id"],
        duration_s=float(len(turns)),
        dataset=dataset,
        metadata={**(metadata or {}), "text_only": True},
        turns=turns,
    )


class Pipeline:
    """Runs the configured modules in order; idempotent (cached results, MERGE writes)."""

    def __init__(self, config: AppConfig | None = None, write_graph: bool = True) -> None:
        self.config = config or get_config()
        self.llm = LLMClient(self.config)
        self.write_graph = write_graph
        self._writer: GraphWriter | None = None
        self._vectors: VectorStore | None = None

    @property
    def writer(self) -> GraphWriter:
        if self._writer is None:
            self._writer = GraphWriter(self.config, GraphClient(self.config))
        return self._writer

    @property
    def vectors(self) -> VectorStore:
        if self._vectors is None:
            self._vectors = VectorStore(self.config)
        return self._vectors

    def process(
        self,
        call: CallTranscript,
        audio: Path | None = None,
        force: bool = False,
        progress: Progress | None = None,
    ) -> CallResult:
        """Analyse one transcript and store results."""
        cfg = self.config
        rp = result_path(cfg, call.call_id)
        if rp.exists() and not force:
            res = CallResult.model_validate_json(rp.read_text())
            if self.write_graph:
                self.writer.write(res)
            return res
        status: dict[str, str] = {"core": "done"}
        notify = progress or (lambda m, s: None)
        write_status(cfg, call.call_id, status)
        safe = redact_call(cfg, call)
        status["privacy"] = "done"
        modules: dict[str, dict[str, Any]] = {}
        context: dict[str, Any] = {"llm": self.llm, "audio_path": audio}
        for name, entry in cfg.pipeline.modules.items():  # type: ignore[attr-defined]
            if not entry.get("enabled", False):
                status[name] = "skipped"
                continue
            if call.metadata.get("text_only") and name in entry.get("skip_for_text", []):
                status[name] = "skipped"
                continue
            status[name] = "running"
            write_status(cfg, call.call_id, status)
            notify(name, "running")
            t0 = time.time()
            try:
                out = MODULES.get(name)(cfg).run(safe, cfg, **context)
                modules[name] = out.model_dump()
                context[name] = out
                status[name] = "done"
            except Exception as exc:  # a failing module must not lose the others' results
                logger.exception("module {} failed on {}", name, call.call_id)
                status[name] = f"failed: {exc}"
            notify(name, status[name])
            logger.info("{} {} {:.1f}s", call.call_id, name, time.time() - t0)
        res = CallResult(transcript=safe, modules=modules, status=status)
        if self.write_graph:
            status["graph"] = "running"
        write_status(cfg, call.call_id, status)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(res.model_dump_json())
        if self.write_graph:
            self.writer.write(res)
            self.vectors.add_call(safe)
            status["graph"] = "done"
        write_status(cfg, call.call_id, status)
        return res

    def process_audio(
        self,
        audio: Path,
        call_id: str,
        dataset: str | None = None,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
        progress: Progress | None = None,
    ) -> CallResult:
        """Full path from audio file."""
        call = run_core(self.config, audio, call_id, dataset, metadata, self.llm)
        return self.process(call, audio=audio, force=force, progress=progress)

    def process_manifest(
        self, dataset: str, limit: int | None = None, force: bool = False
    ) -> list[CallResult]:
        """Process the (audio) rows of a dataset manifest."""
        from callgraph.data.manifest import read_manifest

        man = read_manifest(self.config, dataset)
        man = man[man["audio_path"].notna()]
        out = []
        for _, r in man.head(limit).iterrows():
            meta = {
                k: v
                for k, v in (r["labels"] or {}).items()
                if k in ("wait_time_s", "date", "category")
            }
            out.append(
                self.process_audio(
                    self.config.root / r["audio_path"], str(r["id"]), dataset, meta, force
                )
            )
        if self.write_graph and out:
            self.writer.refresh_resolution_counts()
        return out
