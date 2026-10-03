"""Real-world protocol on UniData/AxonData: WER vs vendor transcripts, intent/emotion vs silver labels, summaries, case studies."""

from __future__ import annotations

import json
import re
from typing import Any

import jiwer
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from callgraph.config import AppConfig
from callgraph.core.preprocess import normalize_for_wer
from callgraph.data.manifest import read_manifest
from callgraph.silver import labels_path, load_real_calls
from eval.common import load_doc, run_dir, save_result


def strip_tags(config: AppConfig, text: str) -> str:
    """Drop bracketed anonymisation/speaker tags before WER."""
    return re.sub(config.evaluation.real_world["strip_pattern"], " ", text)  # type: ignore[attr-defined]


def run(config: AppConfig) -> dict[str, Any]:
    ev = config.evaluation  # type: ignore[attr-defined]
    results = load_real_calls(config, list(ev.silver_labels["real_datasets"]), english_only=False)
    out: dict[str, Any] = {
        "n_calls": len(results),
        "silver_disclaimer": ev.silver_labels["disclaimer"],
    }
    man = read_manifest(config, "unidata").set_index("id")
    wers, summ_h, summ_r, rows = [], [], [], []
    for res in results:
        t = res["transcript"]
        cid = t["call_id"]
        row = {
            "call_id": cid,
            "dataset": t["dataset"],
            "language": t["language"],
            "non_english_flag": t["non_english"],
            "intent": (res["modules"].get("intent") or {}).get("top_intent"),
        }
        if cid in man.index:
            doc = load_doc(config, man.loc[cid])
            ref = normalize_for_wer(strip_tags(config, " ".join(x["text"] for x in doc["turns"])))
            hyp = normalize_for_wer(" ".join(x["text"] for x in t["turns"]))
            row["wer"] = jiwer.wer(ref, hyp) if ref else None
            wers.append(row["wer"])
            row["category"] = man.loc[cid].labels.get("category")
            ai = man.loc[cid].labels.get("ai_summary")
            if ai and res["modules"].get("summary"):
                summ_h.append(res["modules"]["summary"]["summary"])
                summ_r.append(ai)
        rows.append(row)
    table = pd.DataFrame(rows)
    out["wer_vs_unidata_transcripts"] = {
        "mean": float(pd.Series(wers).mean()) if wers else None,
        "n": len(wers),
    }
    out["language_flags"] = table[["call_id", "language", "non_english_flag"]].to_dict("records")
    if summ_h:
        from eval.bertscore import bertscore_f1

        f1 = bertscore_f1(config, summ_h, summ_r)
        out["bertscore_vs_unidata_ai_summaries"] = {
            "f1": float(sum(f1) / len(f1)),
            "n": len(summ_h),
            "note": "UniData AI summaries are NOT gold standard",
        }
    # intent + emotion vs silver
    ip = labels_path(config, "real_call_intent")
    if ip.exists():
        silver = {i["call_id"]: i["final"] for i in json.loads(ip.read_text())["items"]}
        m = table[table.call_id.isin(silver)]
        if len(m):
            out["intent_vs_silver"] = {
                "accuracy": accuracy_score([silver[c] for c in m.call_id], m.intent),
                "n": len(m),
            }
    ep = labels_path(config, "real_call_emotion")
    if ep.exists():
        items = json.loads(ep.read_text())["items"]
        pred = {
            (r["transcript"]["call_id"], e["turn_index"]): e["emotion"]
            for r in results
            for e in (r["modules"].get("emotion") or {"turns": []})["turns"]
        }
        pairs = [
            (i["final"], pred[(i["call_id"], i["turn_index"])])
            for i in items
            if (i["call_id"], i["turn_index"]) in pred
        ]
        if pairs:
            out["emotion_vs_silver"] = {
                "weighted_f1": f1_score(
                    [a for a, _ in pairs], [b for _, b in pairs], average="weighted"
                ),
                "n": len(pairs),
            }
    save_result(config, "real_world", out, table)
    # domain gap (exp 5)
    gap = []
    for name, rec in (
        ("intent_accuracy", ("intent_metrics", "accuracy")),
        ("emotion_weighted_f1", ("emotion_f1", "weighted_f1")),
    ):
        p = run_dir(config) / f"{rec[0]}.json"
        if p.exists():
            rows_ = json.loads(p.read_text())["rows"]
            pub = [
                r[rec[1]]
                for r in rows_
                if r["dataset"] in ("abcd", "meld")
                and r.get("model", r.get("variant")) in ("transformer", "fused")
            ]
            real = (
                out.get("intent_vs_silver", {}).get("accuracy")
                if name.startswith("intent")
                else out.get("emotion_vs_silver", {}).get("weighted_f1")
            )
            gap.append(
                {
                    "metric": name,
                    "public_test": pub[0] if pub else None,
                    "real_calls_silver": real,
                    "drop": (pub[0] - real) if pub and real is not None else None,
                }
            )
    save_result(
        config,
        "exp5_domain_gap",
        {"rows": gap, "disclaimer": ev.silver_labels["disclaimer"]},
        pd.DataFrame(gap),
    )
    # case studies
    d = run_dir(config) / "case_studies"
    d.mkdir(exist_ok=True)
    for res in results[: int(ev.real_world["case_studies"])]:
        m_ = res["modules"]
        lines = [
            f"# Case study: {res['transcript']['call_id']}",
            "",
            f"- dataset: {res['transcript']['dataset']}  |  duration: {res['transcript']['duration_s']:.0f}s",
            f"- intent: {(m_.get('intent') or {}).get('top_intent')}",
            f"- summary: {(m_.get('summary') or {}).get('summary')}",
            f"- extraction: {json.dumps(m_.get('extraction'), ensure_ascii=False)}",
            f"- emotion: start={(m_.get('emotion') or {}).get('start_emotion')} end={(m_.get('emotion') or {}).get('end_emotion')} frustration={(m_.get('emotion') or {}).get('frustration_score')}",
            f"- compliance: {json.dumps((m_.get('compliance') or {}).get('missed_steps'))} score={(m_.get('compliance') or {}).get('score')}",
            f"- dynamics flags: {(m_.get('dynamics') or {}).get('flags')}",
            "",
            "## First turns",
            "",
        ]
        lines += [
            f"- **{x['role']}** [{x['start']:.0f}s] {x['text']}"
            for x in res["transcript"]["turns"][:12]
        ]
        (d / f"{res['transcript']['call_id']}.md").write_text("\n".join(lines))
    return out
