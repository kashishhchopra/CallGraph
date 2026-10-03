"""Audio-first intent evaluation: intent from Whisper transcripts of SpokenWOZ audio vs intent from gold transcripts."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from callgraph.config import AppConfig
from callgraph.intent.data import customer_text
from callgraph.intent.model import IntentClassifier
from callgraph.pipeline import result_path
from eval.common import load_doc, save_result, subset


def run(config: AppConfig) -> dict[str, Any]:
    ev = config.evaluation  # type: ignore[attr-defined]
    n = int(ev.process_subset["spokenwoz"])
    clf = IntentClassifier(config, "transformer")
    cutoffs = [None, *[float(c) for c in ev.intent_cutoffs_s]]
    rows = []
    for _, row in subset(config, "spokenwoz", n).iterrows():
        rp = result_path(config, str(row["id"]))
        if not rp.exists():
            continue
        asr_turns = json.loads(rp.read_text())["transcript"]["turns"]
        gold_turns = load_doc(config, row)["turns"]
        for c in cutoffs:
            for src, turns in (("whisper_asr", asr_turns), ("gold_transcript", gold_turns)):
                text = customer_text(turns, until_s=c) or " "
                rows.append(
                    {
                        "call_id": row["id"],
                        "cutoff_s": "full" if c is None else c,
                        "source": src,
                        "pred": clf.labels[int(clf.proba([text]).argmax())],
                        "label": row["labels"]["intent"],
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"note": "no processed SpokenWOZ calls yet"}
    out = [
        {
            "source": s,
            "cutoff_s": c,
            "n": len(g),
            "accuracy": accuracy_score(g.label, g.pred),
            "macro_f1": f1_score(g.label, g.pred, average="macro"),
        }
        for (s, c), g in df.groupby(["source", "cutoff_s"])
    ]
    save_result(
        config,
        "intent_from_audio",
        {"rows": out, "note": "labels are domain-derived; ASR = Whisper on the audio"},
        pd.DataFrame(out),
    )
    return {"rows": out}
