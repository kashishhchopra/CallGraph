"""RAG comparison (headline experiment 7): Vector vs Graph vs Hybrid on the question set, by question type."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from pydantic import BaseModel

from callgraph.config import AppConfig
from callgraph.llm.client import LLMClient, LLMError, render_prompt
from callgraph.rag.strategies import get_strategy
from callgraph.silver import labels_path
from eval.common import save_result


class _Judge(BaseModel):
    correct: bool
    reason: str = ""


def exact_match(answer: str, q: dict[str, Any]) -> bool:
    """All reference key values (first column(s)) appear in the answer; counts compare as numbers."""
    a = answer.lower()
    vals = [
        str(r[k]).lower() for r in q["reference_rows"] for k in q["keys"] if r.get(k) is not None
    ]
    if len(q["reference_rows"]) == 1 and q["keys"] == ["calls"]:
        return bool(re.search(rf"\b{vals[0]}\b", a))
    return bool(vals) and all(v in a for v in vals)


def llm_judge(llm: LLMClient, config: AppConfig, q: dict[str, Any], answer: str) -> bool:
    try:
        return bool(llm.complete_model(render_prompt("judge", config, question=q["question"], reference=q["reference_text"], answer=answer), _Judge).correct)  # type: ignore[attr-defined]
    except LLMError:
        return False


def ragas_scores(config: AppConfig, rows: list[dict[str, Any]]) -> dict[str, float]:
    """Faithfulness + context precision. Uses RAGAS if importable and usable, else an LLM-judge implementation."""
    from eval.ragas_metrics import score

    return score(config, rows)


def run(config: AppConfig) -> dict[str, Any]:
    p = labels_path(config, "graphrag_questions")
    if not p.exists():
        return {"note": "build questions first (eval.questions.build)"}
    qs = json.loads(p.read_text())["questions"][: int(config.evaluation.subsets["rag_questions"])]  # type: ignore[attr-defined]
    judge_llm = LLMClient(config)
    rows, per_q = [], []
    for name in config.evaluation.rag["strategies"]:  # type: ignore[attr-defined]
        strat = get_strategy(config, name)
        recs = []
        for q in qs:
            r = strat.ask(q["question"])
            rec = {
                "strategy": name,
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "answer": r.answer,
                "latency_s": r.latency_s,
                "exact": exact_match(r.answer, q),
                "llm_judge": llm_judge(judge_llm, config, q, r.answer),
                "contexts": r.contexts,
                "reference": q["reference_text"],
                "cypher": r.cypher,
                "fallback": r.fallback_used,
            }
            recs.append(rec)
        per_q += recs
        rg = ragas_scores(config, recs)
        for qt in ["all", *config.evaluation.rag["question_types"]]:  # type: ignore[attr-defined]
            sub = [x for x in recs if qt == "all" or x["type"] == qt]
            if sub:
                rows.append(
                    {
                        "strategy": name,
                        "question_type": qt,
                        "n": len(sub),
                        "exact_match": sum(x["exact"] for x in sub) / len(sub),
                        "llm_judge_acc": sum(x["llm_judge"] for x in sub) / len(sub),
                        "latency_s": sum(x["latency_s"] for x in sub) / len(sub),
                        **(
                            {
                                "faithfulness": rg.get("faithfulness"),
                                "context_precision": rg.get("context_precision"),
                            }
                            if qt == "all"
                            else {}
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    save_result(config, "exp7_rag_comparison", {"rows": rows}, df)
    save_result(
        config,
        "rag_per_question",
        {"records": per_q},
        pd.DataFrame(per_q).drop(columns=["contexts"]),
    )
    return {"rows": rows}
