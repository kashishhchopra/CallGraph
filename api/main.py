"""CallGraph REST API."""

from __future__ import annotations

import json
import shutil
import uuid
from functools import lru_cache
from http import HTTPStatus
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from api import results as res_store
from callgraph.config import get_config
from callgraph.graph.client import GraphClient
from callgraph.llm.client import LLMClient
from callgraph.logging_setup import setup_logging
from callgraph.pipeline import Pipeline, call_dir, result_path, write_status
from callgraph.rag.strategies import get_strategy

setup_logging()
cfg = get_config()
app = FastAPI(
    title="CallGraph API",
    version="0.1.0",
    description="Call-center intelligence: results, analytics, GraphRAG.",
)


class AskRequest(BaseModel):
    """Body of POST /ask."""

    question: str
    strategy: str = "graph"


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    return error(exc.status_code, "http_error", str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return error(
        HTTPStatus.UNPROCESSABLE_ENTITY, "validation_error", json.dumps(exc.errors(), default=str)
    )


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    return error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", f"{type(exc).__name__}: {exc}")


@lru_cache(maxsize=1)
def pipeline() -> Pipeline:
    return Pipeline(cfg)


@lru_cache(maxsize=1)
def graph() -> GraphClient:
    return GraphClient(cfg)


def upload_dir() -> Path:
    d = cfg.path("raw_dir") / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_job(audio: Path, call_id: str) -> None:
    write_status(cfg, call_id, {"core": "running"})
    pipeline().process_audio(audio, call_id, dataset="upload", force=True)


@app.on_event("startup")
def warm_up() -> None:
    """Load the embedding model, vector index and LLM in the background so the first question is not slow."""
    import threading

    def work() -> None:
        try:
            from callgraph.rag.vector_store import VectorStore

            VectorStore(cfg).search(str(cfg.rag.model_extra["warmup_query"]), 1)  # type: ignore[index]
            LLMClient(cfg).warm()
        except Exception as exc:  # warm-up is best effort
            logger.warning("warm-up skipped: {}", exc)

    threading.Thread(target=work, daemon=True).start()


@app.post("/calls", status_code=HTTPStatus.ACCEPTED, tags=["calls"])
async def upload_call(background: BackgroundTasks, file: UploadFile = File(...)) -> dict[str, str]:
    """Upload audio and start processing in the background."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in set(cfg.pipeline.audio["audio_extensions"]):  # type: ignore[attr-defined]
        raise HTTPException(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, f"Unsupported audio type '{ext}'")
    call_id = f"upload_{uuid.uuid4().hex[: int(cfg.api.call_id_hex_len)]}"
    dest = upload_dir() / f"{call_id}{ext}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    write_status(cfg, call_id, {"core": "queued"})
    background.add_task(run_job, dest, call_id)
    return {"call_id": call_id, "status": "queued"}


def load_result(call_id: str) -> dict[str, Any]:
    p = result_path(cfg, call_id)
    if not p.exists():
        raise HTTPException(HTTPStatus.NOT_FOUND, f"Call '{call_id}' not found or still processing")
    return json.loads(p.read_text())


@app.get("/calls/{call_id}", tags=["calls"])
def get_call(call_id: str) -> dict[str, Any]:
    """Full results (transcript + module outputs) for one call."""
    return load_result(call_id)


@app.get("/calls/{call_id}/status", tags=["calls"])
def call_status(call_id: str) -> dict[str, Any]:
    """Per-module pipeline progress."""
    p = call_dir(cfg, call_id) / "status.json"
    if not p.exists():
        raise HTTPException(HTTPStatus.NOT_FOUND, f"No status for '{call_id}'")
    status = json.loads(p.read_text())
    finished = result_path(cfg, call_id).exists() and not any(
        v in ("queued", "running") for v in status.values()
    )
    return {"call_id": call_id, "status": status, "finished": finished}


@app.get("/calls/{call_id}/audio", tags=["calls"])
def call_audio(call_id: str) -> FileResponse:
    """Audio file (uploads or manifest audio) for the call viewer."""
    path = res_store.find_audio(cfg, call_id, upload_dir())
    if path is None:
        raise HTTPException(HTTPStatus.NOT_FOUND, "No audio for this call")
    return FileResponse(path)


@app.get("/calls", tags=["calls"])
def list_calls(
    page: int = Query(1, ge=1),
    page_size: int = Query(cfg.api.default_page_size, ge=1, le=cfg.api.max_page_size),
    dataset: str | None = None,
    intent: str | None = None,
    outcome: str | None = None,
    emotion: str | None = None,
    q: str | None = None,
    min_frustration: float | None = None,
) -> dict[str, Any]:
    """Paginated, filterable, searchable list of processed calls (newest first)."""
    rows = [res_store.summary_row(r) for r in res_store.iter_results(cfg)]
    needle = q.lower() if q else None

    def keep(r: dict[str, Any]) -> bool:
        if dataset and r["dataset"] != dataset or intent and r["intent"] != intent:
            return False
        if outcome and r["outcome"] != outcome or emotion and r["end_emotion"] != emotion:
            return False
        if min_frustration is not None and (r["frustration_score"] or 0) < min_frustration:
            return False
        blob = " ".join(
            str(r.get(k) or "") for k in ("call_id", "issue", "summary", "intent")
        ).lower()
        return needle is None or needle in blob

    rows = sorted((r for r in rows if keep(r)), key=lambda r: r["processed_at"], reverse=True)
    start = (page - 1) * page_size
    return {
        "total": len(rows),
        "page": page,
        "page_size": page_size,
        "items": rows[start : start + page_size],
    }


@app.get("/overview", tags=["analytics"])
def overview(days: int | None = Query(None, ge=1)) -> dict[str, Any]:
    """KPI tiles, daily volume, intent mix and recent calls for the overview page."""
    ui = cfg.dashboard_ui.model_extra or {}
    return res_store.overview(cfg, days or int(ui["default_range_days"]))


@app.get("/analytics/{metric}", tags=["analytics"])
def analytics(metric: str) -> dict[str, Any]:
    """Aggregates for dashboard charts (Cypher defined in configs/analytics.yaml)."""
    metrics = dict(cfg.analytics.metrics) if hasattr(cfg.analytics, "metrics") else {}  # type: ignore[attr-defined]
    if metric not in metrics:
        raise HTTPException(
            HTTPStatus.NOT_FOUND, f"Unknown metric '{metric}'. Available: {sorted(metrics)}"
        )
    spec = metrics[metric]
    if spec.get("source") == "results":
        return {"metric": metric, "rows": res_store.dynamics_means(cfg, list(spec["features"]))}
    return {
        "metric": metric,
        "rows": graph().ro_query(spec["cypher"], dict(spec.get("params", {}))),
    }


@app.post("/ask", tags=["rag"])
def ask(req: AskRequest) -> dict[str, Any]:
    """Answer a question with Vector, Graph or Hybrid RAG."""
    if req.strategy not in list(cfg.rag.model_extra["strategies"]):  # type: ignore[attr-defined]
        raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, f"strategy must be one of {list(cfg.rag.model_extra['strategies'])}")  # type: ignore[attr-defined]
    return get_strategy(cfg, req.strategy, graph=graph()).ask(req.question).model_dump()


@app.get("/graph/subgraph", tags=["graph"])
def subgraph(
    calls: bool = False,
    issue: str | None = None,
    outcome: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Issue -> action -> outcome subgraph for the explorer."""
    spec = cfg.analytics.subgraph  # type: ignore[attr-defined]
    if calls:
        if not issue:
            raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, "calls=true requires an issue")
        rows = graph().ro_query(
            spec["calls_cypher"], {"issue": issue, "limit": int(spec["calls_limit"])}
        )
        return {
            "nodes": [
                {"id": f"call:{r['call_id']}", "label": r["call_id"], "type": "Call"} for r in rows
            ],
            "edges": [
                {"source": f"call:{r['call_id']}", "target": f"issue:{issue}", "weight": 1}
                for r in rows
            ],
        }
    lim = limit or int(spec["default_limit"])
    rows = graph().ro_query(spec["cypher"], {"issue": issue, "outcome": outcome, "limit": lim})
    nodes: dict[str, dict[str, str]] = {}
    edges: list[dict[str, Any]] = []
    for r in rows:
        ids = {
            "Issue": f"issue:{r['issue']}",
            "Action": f"action:{r['action']}",
            "Outcome": f"outcome:{r['outcome']}",
        }
        nodes[ids["Issue"]] = {"id": ids["Issue"], "label": r["issue"], "type": "Issue"}
        nodes[ids["Action"]] = {"id": ids["Action"], "label": r["action"], "type": "Action"}
        nodes[ids["Outcome"]] = {"id": ids["Outcome"], "label": r["outcome"], "type": "Outcome"}
        edges.append({"source": ids["Issue"], "target": ids["Action"], "weight": r["n"]})
        edges.append({"source": ids["Action"], "target": ids["Outcome"], "weight": r["n"]})
    return {"nodes": list(nodes.values()), "edges": edges}


@app.get("/config/public", tags=["config"])
def public_config() -> dict[str, Any]:
    """Everything the UI needs: navigation, colours, label sets, questions and pipeline stages."""
    ui = dict(cfg.dashboard_ui.model_extra or {})
    modules = list(cfg.pipeline.modules.keys())  # type: ignore[attr-defined]
    return {
        **ui,
        "strategies": list(cfg.rag.model_extra["strategies"]),  # type: ignore[index]
        "intents": list(cfg.intents.model_extra["shared_labels"]),  # type: ignore[index]
        "emotions": list(cfg.emotions.model_extra["shared_labels"]),  # type: ignore[index]
        "outcomes": list(cfg.models.extraction["outcomes"]),  # type: ignore[attr-defined]
        "rubric_items": [i["id"] for i in cfg.rubric.items],  # type: ignore[attr-defined]
        "compliance_scale": cfg.rubric.scoring["scale"],  # type: ignore[attr-defined]
        "compliance_na_marker": cfg.rubric.scoring["na_evidence"],  # type: ignore[attr-defined]
        "pipeline_stages": ["core", "privacy", *modules, "graph"],
        "audio_extensions": list(cfg.pipeline.audio["audio_extensions"]),  # type: ignore[attr-defined]
        "bytes_per_mb": cfg.downloads.bytes_per_mb,  # type: ignore[attr-defined]
    }


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    """Health of FalkorDB, the LLM endpoint and trained models."""
    models = {
        k: (cfg.root / getattr(cfg.models, k)["output_dir"]).exists()
        for k in ("intent", "emotion_audio", "emotion_text", "summarization")
    }
    return {"falkordb": graph().healthy(), "llm": LLMClient(cfg).health(), "models": models}


class NoCacheStaticFiles(StaticFiles):
    """Static files that browsers must revalidate, so a rebuilt UI is never served from a stale cache."""

    async def get_response(self, path: str, scope: Any) -> Any:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


FRONTEND_DIR = cfg.root / "frontend"
if FRONTEND_DIR.is_dir():  # mounted last so every API route above takes precedence
    app.mount("/", NoCacheStaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
