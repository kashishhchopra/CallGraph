"""Uncached RAG latency benchmark: a small stratified question sample per strategy (run with LLM__CACHE_ENABLED=false)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from callgraph.config import AppConfig
from callgraph.rag.strategies import get_strategy
from callgraph.silver import labels_path
from eval.common import save_result


def run(config: AppConfig) -> dict[str, Any]:
    if config.llm.cache_enabled:  # type: ignore[attr-defined]
        return {
            "note": "refusing to benchmark latency with the LLM cache enabled (set LLM__CACHE_ENABLED=false)"
        }
    ev = config.evaluation  # type: ignore[attr-defined]
    qs = json.loads(labels_path(config, "graphrag_questions").read_text())["questions"]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for q in qs:
        by_type.setdefault(q["type"], []).append(q)
    sample: list[dict[str, Any]] = []
    i = 0
    while len(sample) < int(ev.rag["latency_questions"]) and any(by_type.values()):
        t = list(by_type)[i % len(by_type)]
        if by_type[t]:
            sample.append(by_type[t].pop(0))
        i += 1
    rows = []
    for name in ev.rag["strategies"]:
        strat = get_strategy(config, name)
        for q in sample:
            r = strat.ask(q["question"])
            rows.append({"strategy": name, "question_type": q["type"], "latency_s": r.latency_s})
    df = pd.DataFrame(rows)
    out = df.groupby(["strategy", "question_type"], as_index=False).agg(
        n=("latency_s", "size"), mean_latency_s=("latency_s", "mean")
    )
    overall = df.groupby("strategy", as_index=False).agg(
        n=("latency_s", "size"), mean_latency_s=("latency_s", "mean")
    )
    overall.insert(1, "question_type", "all")
    table = pd.concat([overall, out], ignore_index=True)
    save_result(
        config,
        "rag_latency",
        {
            "rows": table.to_dict("records"),
            "note": "LLM cache disabled; local 3B model on a loaded 8 GB CPU",
        },
        table,
    )
    return {"rows": len(table)}
