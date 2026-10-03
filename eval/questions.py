"""Build the GraphRAG question set: templated questions whose reference answers are EXECUTED on the graph."""

from __future__ import annotations

import json
import random
from typing import Any

from loguru import logger

from callgraph.config import AppConfig
from callgraph.graph.client import GraphClient
from callgraph.silver import labels_path


def reference_text(rows: list[dict[str, Any]]) -> str:
    return "; ".join(", ".join(f"{k}={v}" for k, v in r.items()) for r in rows)


def build(config: AppConfig, graph: GraphClient | None = None) -> dict[str, Any]:
    g = graph or GraphClient(config)
    spec = dict(config.rag_questions.model_extra or {})  # type: ignore[attr-defined]
    rng = random.Random(int(spec["seed"]))
    out: list[dict[str, Any]] = []
    for tpl in spec["templates"]:
        if "fixed" in tpl:
            combos = list(tpl["fixed"])
        else:
            pool = g.ro_query(tpl["sample"]["cypher"])
            rng.shuffle(pool)
            combos = pool[: int(spec["per_template"])]
        for i, params in enumerate(combos):
            rows = g.ro_query(tpl["answer"]["cypher"], dict(params))
            if not rows:
                continue
            variant = tpl["variants"][i % len(tpl["variants"])]
            out.append(
                {
                    "id": f"{tpl['id']}_{i}",
                    "type": tpl["type"],
                    "template": tpl["id"],
                    "question": variant.format(**params),
                    "params": params,
                    "reference_rows": rows,
                    "reference_text": reference_text(rows),
                    "keys": tpl["answer"]["keys"],
                }
            )
    logger.info(
        "Built {} questions: {}",
        len(out),
        {t: sum(1 for q in out if q["type"] == t) for t in {q["type"] for q in out}},
    )
    data = {
        "note": "Questions are templated; reference answers are the executed Cypher results on the processed-call graph.",
        "questions": out,
    }
    p = labels_path(config, "graphrag_questions")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1, default=str))
    return data
