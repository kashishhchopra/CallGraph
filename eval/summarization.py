"""Summarization evaluation: ROUGE-1/2/L + BERTScore on DialogSum, TweetSumm, AMI; Lead-3 baseline; human-rating sheet."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import BaseModel
from rouge_score import rouge_scorer

from callgraph.config import AppConfig
from callgraph.llm.client import LLMClient, LLMError, render_prompt
from callgraph.summarization.summarize import summarize_text_seq2seq
from callgraph.summarization.train import dialog_text
from eval.common import run_dir, save_result, subset


class _Reply(BaseModel):
    summary: str


def lead_k(config: AppConfig, doc: dict[str, Any]) -> str:
    k = int(config.evaluation.summarization["lead_k_turns"])  # type: ignore[attr-defined]
    return " ".join(t["text"] for t in doc["turns"][:k])


def gather(config: AppConfig) -> list[dict[str, Any]]:
    """(dataset, id, dialogue text, doc, reference) items."""
    ev = config.evaluation  # type: ignore[attr-defined]
    items = []
    for ds, n in ev.subsets["summarization"].items():
        man = subset(config, ds, int(n), "test" if ds != "ami" else None)
        for _, row in man.iterrows():
            doc = json.loads((config.root / row["transcript_path"]).read_text())
            ref = row["labels"].get("summary") or doc.get("labels", {}).get("summary") or ""
            if not ref:
                continue
            text = dialog_text(config, row["transcript_path"])
            if ds == "ami":
                text = text[: int(ev.summarization["ami_max_chars"])]
            items.append({"dataset": ds, "id": row["id"], "text": text, "doc": doc, "ref": ref})
    return items


def run(config: AppConfig) -> dict[str, Any]:
    from eval.bertscore import bertscore_f1

    ev = config.evaluation  # type: ignore[attr-defined]
    llm = LLMClient(config)
    items = gather(config)
    systems = list(ev.summarization["systems"])
    outputs: dict[str, list[str]] = {s: [] for s in systems}
    for it in items:
        for s in systems:
            if s == "lead3":
                outputs[s].append(lead_k(config, it["doc"]))
            elif s == "llm":
                try:
                    outputs[s].append(llm.complete_model(render_prompt("summary_plain", config, transcript=it["text"]), _Reply).summary)  # type: ignore[attr-defined]
                except LLMError:
                    outputs[s].append("")
            elif s == "seq2seq":
                try:
                    outputs[s].append(summarize_text_seq2seq(config, it["text"]))
                except OSError:
                    outputs[s].append("")
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    for s in systems:
        if not any(outputs[s]):
            rows.append({"dataset": "*", "system": s, "note": "not run (model missing)"})
            continue
        for ds in sorted({i["dataset"] for i in items}):
            idx = [k for k, i in enumerate(items) if i["dataset"] == ds]
            refs, hyps = [items[k]["ref"] for k in idx], [outputs[s][k] for k in idx]
            r = pd.DataFrame(
                [
                    {m: v.fmeasure for m, v in sc.score(a, b).items()}
                    for a, b in zip(refs, hyps, strict=False)
                ]
            ).mean()
            f1 = bertscore_f1(config, hyps, refs)
            rows.append(
                {
                    "dataset": ds,
                    "system": s,
                    "n": len(idx),
                    "rouge1": r.rouge1,
                    "rouge2": r.rouge2,
                    "rougeL": r.rougeL,
                    "bertscore_f1": float(sum(f1) / len(f1)),
                }
            )
    save_result(config, "summarization", {"rows": rows}, pd.DataFrame(rows))
    n = int(ev.summarization["human_sheet_n"])
    sheet = (
        pd.DataFrame(
            [
                {
                    "dataset": it["dataset"],
                    "id": it["id"],
                    "reference": it["ref"],
                    **{f"{s}_summary": outputs[s][k] for s in systems},
                    "fluency_1to5": "",
                    "faithfulness_1to5": "",
                }
                for k, it in enumerate(items)
            ]
        )
        .sample(frac=1, random_state=int(config.split.seed))
        .head(n)
    )  # type: ignore[attr-defined]
    sheet.to_csv(run_dir(config) / "human_rating_sheet.csv", index=False)
    return {"rows": rows}
