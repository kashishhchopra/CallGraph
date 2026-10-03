"""Compliance evaluation: step-level F1 on ABCD (required actions), keyword baseline, real-call agreement."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

from callgraph.config import AppConfig
from callgraph.llm.client import LLMClient, LLMError, render_prompt
from callgraph.silver import labels_path
from eval.common import save_result, subset


class _Reply(BaseModel):
    performed: dict[str, bool]


def keyword_hits(action: str, agent_text: str) -> bool:
    """Baseline: all words of the action name appear in agent speech."""
    return all(w in agent_text for w in action.split("-"))


def run(config: AppConfig) -> dict[str, Any]:
    llm = LLMClient(config)
    ev = config.evaluation  # type: ignore[attr-defined]
    man = subset(config, "abcd", int(ev.compliance["abcd_subset"]), "test")
    y, p_llm, p_kw = [], [], []
    for _, row in man.iterrows():
        doc = json.loads((config.root / row["transcript_path"]).read_text())
        req = [a for a in row["labels"]["required_actions"] if a]
        if not req:
            continue
        transcript = "\n".join(f"{t['role'].upper()}: {t['text']}" for t in doc["turns"])
        agent = " ".join(t["text"].lower() for t in doc["turns"] if t["role"] == "agent")
        try:
            got = llm.complete_model(render_prompt("abcd_actions", config, transcript=transcript, actions=req), _Reply).performed  # type: ignore[attr-defined]
        except LLMError:
            got = {}
        for a in req:
            y.append(a in row["labels"]["performed_actions"])
            p_llm.append(bool(got.get(a, False)))
            p_kw.append(keyword_hits(a, agent))
    rows = [
        {
            "system": name,
            "n_steps": len(y),
            "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0),
        }
        for name, p in (("llm_rubric", p_llm), ("keyword_rules", p_kw))
    ]
    out: dict[str, Any] = {"abcd": rows}
    real = labels_path(config, "real_call_compliance")
    if real.exists():
        from callgraph.compliance.audit import audit_keywords, audit_llm
        from callgraph.schemas import CallTranscript
        from callgraph.silver import load_real_calls

        silver = {
            (i["call_id"], i["rubric_item"]): bool(i["final"])
            for i in json.loads(real.read_text())["items"]
        }
        cmp_rows = []
        for res in load_real_calls(config, list(ev.silver_labels["real_datasets"])):
            t = CallTranscript.model_validate(res["transcript"])
            upset = bool((res["modules"].get("emotion") or {}).get("ends_negative"))
            for name, o in (
                (
                    "llm_rubric",
                    audit_llm(
                        config,
                        t,
                        (res["modules"].get("intent") or {}).get("top_intent", "unknown"),
                        upset,
                        llm,
                    ),
                ),
                ("keyword_rules", audit_keywords(config, t, upset)),
            ):
                for it in o.items:
                    if (t.call_id, it.rubric_item) in silver:
                        cmp_rows.append(
                            {
                                "system": name,
                                "call_id": t.call_id,
                                "item": it.rubric_item,
                                "pred": it.passed,
                                "silver": silver[(t.call_id, it.rubric_item)],
                            }
                        )
        df = pd.DataFrame(cmp_rows)
        agree = (
            [
                {
                    "system": n,
                    "n": len(g),
                    "agreement": float((g.pred == g.silver).mean()),
                    "kappa_vs_silver": (
                        float(cohen_kappa_score(g.pred, g.silver))
                        if g.pred.nunique() > 1 or g.silver.nunique() > 1
                        else 1.0
                    ),
                }
                for n, g in df.groupby("system")
            ]
            if len(df)
            else []
        )
        out["real_call_agreement_with_silver"] = agree
        out["silver_disclaimer"] = ev.silver_labels["disclaimer"]
    save_result(config, "compliance", out, pd.DataFrame(rows))
    return out
