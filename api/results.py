"""Read processed-call results from disk (cached by file mtime) and derive summary rows and aggregates."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from callgraph.config import AppConfig

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def iter_results(cfg: AppConfig) -> list[dict[str, Any]]:
    """All processed results, each with a ``processed_at`` ISO timestamp (file mtime)."""
    out: list[dict[str, Any]] = []
    for p in sorted((cfg.path("processed_dir") / "calls").glob("*/result.json")):
        mtime = p.stat().st_mtime
        hit = _CACHE.get(str(p))
        if hit is None or hit[0] != mtime:
            res = json.loads(p.read_text())
            res["processed_at"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            _CACHE[str(p)] = (mtime, res)
        out.append(_CACHE[str(p)][1])
    return out


def summary_row(res: dict[str, Any]) -> dict[str, Any]:
    t, m = res["transcript"], res["modules"]
    emo, ext, comp, intent = (
        m.get(k) or {} for k in ("emotion", "extraction", "compliance", "intent")
    )
    return {
        "call_id": t["call_id"],
        "dataset": t.get("dataset"),
        "duration_s": t["duration_s"],
        "language": t.get("language"),
        "intent": intent.get("top_intent"),
        "outcome": ext.get("outcome"),
        "issue": ext.get("issue_type"),
        "summary": (m.get("summary") or {}).get("summary"),
        "start_emotion": emo.get("start_emotion"),
        "end_emotion": emo.get("end_emotion"),
        "frustration_score": emo.get("frustration_score"),
        "compliance_score": comp.get("score"),
        "ends_negative": emo.get("ends_negative"),
        "processed_at": res.get("processed_at"),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def overview(cfg: AppConfig, days: int) -> dict[str, Any]:
    """KPI tiles, daily volume (zero-filled over ``days``), intent mix and recent calls."""
    rows = [summary_row(r) for r in iter_results(cfg)]
    since = datetime.now(timezone.utc) - timedelta(days=days)
    in_range = [r for r in rows if datetime.fromisoformat(r["processed_at"]) >= since]
    per_day: dict[str, int] = {}
    for r in in_range:
        d = r["processed_at"][:10]
        per_day[d] = per_day.get(d, 0) + 1
    today = datetime.now(timezone.utc).date()
    volume = [
        {
            "date": str(today - timedelta(days=i)),
            "count": per_day.get(str(today - timedelta(days=i)), 0),
        }
        for i in range(days - 1, -1, -1)
    ]
    intents: dict[str, int] = {}
    for r in in_range:
        if r["intent"]:
            intents[r["intent"]] = intents.get(r["intent"], 0) + 1
    scale = float(cfg.rubric.scoring["scale"])  # type: ignore[attr-defined]
    return {
        "days": days,
        "kpis": {
            "calls_processed": len(in_range),
            "avg_frustration": _mean(
                [r["frustration_score"] for r in in_range if r["frustration_score"] is not None]
            ),
            "avg_compliance": _mean(
                [r["compliance_score"] for r in in_range if r["compliance_score"] is not None]
            ),
            "compliance_scale": scale,
            "avg_handle_time_s": _mean([r["duration_s"] for r in in_range]),
        },
        "volume": volume,
        "intents": sorted(
            ({"label": k, "value": v} for k, v in intents.items()), key=lambda x: -x["value"]
        ),
        "recent": sorted(in_range, key=lambda r: r["processed_at"], reverse=True)[: int(cfg.dashboard_ui.model_extra["recent_calls"])],  # type: ignore[index]
    }


def dynamics_means(cfg: AppConfig, features: list[str]) -> list[dict[str, Any]]:
    """Mean of selected dynamics features across processed calls that have them."""
    vals: dict[str, list[float]] = {f: [] for f in features}
    for res in iter_results(cfg):
        feats = (res["modules"].get("dynamics") or {}).get("features") or {}
        for f in features:
            if f in feats:
                vals[f].append(float(feats[f]))
    return [{"label": f, "value": _mean(v)} for f, v in vals.items() if v]


def find_audio(cfg: AppConfig, call_id: str, upload_dir: Path) -> Path | None:
    """Audio for a call: an uploaded file, else the manifest's audio path."""
    for p in upload_dir.glob(f"{call_id}.*"):
        return p
    import pandas as pd

    for csv in cfg.path("manifests_dir").glob("*.csv"):
        df = pd.read_csv(csv)
        hit = df[df["id"].astype(str) == call_id]
        if len(hit) and isinstance(hit.iloc[0]["audio_path"], str):
            return cfg.root / hit.iloc[0]["audio_path"]
    return None
