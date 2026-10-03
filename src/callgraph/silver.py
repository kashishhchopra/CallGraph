"""LLM-annotated SILVER labels: two independent annotators, tie-break pass, Cohen's kappa.

These are NOT human gold labels; every output file carries the disclaimer from ``configs/evaluation.yaml``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel
from sklearn.metrics import cohen_kappa_score

from callgraph.config import AppConfig
from callgraph.data.labelmaps import shared_labels
from callgraph.extraction.extract import format_transcript
from callgraph.llm.client import LLMClient, LLMError, render_prompt
from callgraph.schemas import CallTranscript


class _EmotionReply(BaseModel):
    labels: list[dict[str, Any]]


class _ComplianceReply(BaseModel):
    items: list[dict[str, Any]]


class _Verdicts(BaseModel):
    verdicts: dict[str, bool]


class _IntentReply(BaseModel):
    intent: str


class _Label(BaseModel):
    label: str


def labels_path(config: AppConfig, name: str) -> Path:
    return config.path("labels_dir") / f"{name}.json"


def load_real_calls(
    config: AppConfig, datasets: list[str], english_only: bool | None = None
) -> list[dict[str, Any]]:
    """Processed results (result.json) of calls from the given datasets."""
    out = []
    for p in sorted((config.path("processed_dir") / "calls").glob("*/result.json")):
        res = json.loads(p.read_text())
        t = res["transcript"]
        if t.get("dataset") in datasets and not (
            (config.evaluation.silver_labels["english_only"] if english_only is None else english_only)  # type: ignore[attr-defined]
            and t.get("non_english")
        ):
            out.append(res)
    return out


def annotators(config: AppConfig) -> tuple[LLMClient, LLMClient, LLMClient]:
    s = config.evaluation.silver_labels  # type: ignore[attr-defined]
    a, b = s["annotator_models"]
    return LLMClient(config, a), LLMClient(config, b), LLMClient(config, s["tiebreak_model"])


def finish(
    config: AppConfig, name: str, items: list[dict[str, Any]], extra: dict[str, Any]
) -> dict[str, Any]:
    """Compute kappa over items with both labels, write the label file with disclaimer."""
    both = [i for i in items if i.get("a") is not None and i.get("b") is not None]
    kappa = (
        cohen_kappa_score([str(i["a"]) for i in both], [str(i["b"]) for i in both])
        if len({str(i["a"]) for i in both} | {str(i["b"]) for i in both}) > 1
        else 1.0
    )
    out = {
        "disclaimer": config.evaluation.silver_labels["disclaimer"],  # type: ignore[attr-defined]
        "annotator_models": list(config.evaluation.silver_labels["annotator_models"]),  # type: ignore[attr-defined]
        "tiebreak_model": config.evaluation.silver_labels["tiebreak_model"],  # type: ignore[attr-defined]
        "cohen_kappa": kappa,
        "n_items": len(items),
        "n_disagreements": sum(1 for i in both if i["a"] != i["b"]),
        **extra,
        "items": items,
    }
    p = labels_path(config, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str))
    logger.info("{}: {} items, kappa={:.3f}, file={}", name, len(items), kappa, p)
    return out


def tiebreak(
    llm: LLMClient, config: AppConfig, task: str, context: str, a: Any, b: Any, allowed: list[str]
) -> str:
    try:
        r = llm.complete_model(
            render_prompt(
                "tiebreak", config, task=task, context=context, a=a, b=b, allowed=allowed
            ),
            _Label,
        )
        return r.label if r.label in allowed else str(a)  # type: ignore[attr-defined]
    except LLMError:
        return str(a)


def annotate_emotion(config: AppConfig) -> dict[str, Any]:
    labels = shared_labels(config, "emotions")
    A, B, T = annotators(config)
    items = []
    for res in load_real_calls(config, list(config.evaluation.silver_labels["real_datasets"])):  # type: ignore[attr-defined]
        t = CallTranscript.model_validate(res["transcript"])
        turns = [{"index": i, "role": x.role, "text": x.text} for i, x in enumerate(t.turns)]
        win = int(config.evaluation.silver_labels["turn_window"])  # type: ignore[attr-defined]
        got: dict[str, dict[int, str]] = {"a": {}, "b": {}}
        for key, llm in (("a", A), ("b", B)):
            for start in range(0, len(turns), win):
                try:
                    r = llm.complete_model(
                        render_prompt(
                            "annotate_emotion",
                            config,
                            labels=labels,
                            turns=turns[start : start + win],
                        ),
                        _EmotionReply,
                    )
                    got[key].update({int(x["index"]): x["emotion"] for x in r.labels if x.get("emotion") in labels})  # type: ignore[attr-defined]
                except (LLMError, KeyError, ValueError, TypeError):
                    continue
        ctx = int(config.evaluation.silver_labels["tiebreak_context_turns"])  # type: ignore[attr-defined]
        for i, x in enumerate(t.turns):
            if x.role != "customer" or not x.text.strip():
                continue
            a, b = got["a"].get(i), got["b"].get(i)
            final = (
                a
                if a == b and a
                else (
                    tiebreak(
                        T,
                        config,
                        "emotion of a customer turn",
                        "\n".join(
                            f"{y.role.upper()}: {y.text}"
                            for y in t.turns[max(0, i - ctx) : i + ctx + 1]
                        )
                        + f"\n(Disputed turn: {x.text})",
                        a,
                        b,
                        labels,
                    )
                    if (a or b)
                    else None
                )
            )
            if final:
                items.append(
                    {
                        "call_id": t.call_id,
                        "turn_index": i,
                        "text": x.text,
                        "a": a,
                        "b": b,
                        "final": final,
                        "tiebreak_used": a != b,
                    }
                )
    return finish(config, "real_call_emotion", items, {})


def annotate_compliance(config: AppConfig) -> dict[str, Any]:
    from callgraph.compliance.audit import rubric_items

    rub = [r["id"] for r in rubric_items(config)]
    A, B, T = annotators(config)
    items = []
    for res in load_real_calls(config, list(config.evaluation.silver_labels["real_datasets"])):  # type: ignore[attr-defined]
        t = CallTranscript.model_validate(res["transcript"])
        got: dict[str, dict[str, bool]] = {}
        for key, llm in (("a", A), ("b", B)):
            try:
                r = llm.complete_model(
                    render_prompt(
                        "annotate_compliance",
                        config,
                        rubric=rubric_items(config),
                        transcript=format_transcript(t),
                    ),
                    _ComplianceReply,
                )
                got[key] = {x["rubric_item"]: bool(x["passed"]) for x in r.items}  # type: ignore[attr-defined]
            except LLMError:
                got[key] = {}
        for item in rub:
            a, b = got["a"].get(item), got["b"].get(item)
            if a is None and b is None:
                continue
            final = (
                a
                if a == b
                else tiebreak(
                    T,
                    config,
                    f"did the agent satisfy rubric item '{item}'",
                    format_transcript(t),
                    a,
                    b,
                    ["true", "false"],
                )
                == "true"
            )
            items.append(
                {
                    "call_id": t.call_id,
                    "rubric_item": item,
                    "a": a,
                    "b": b,
                    "final": final,
                    "tiebreak_used": a != b,
                }
            )
    return finish(config, "real_call_compliance", items, {})


def annotate_intent(config: AppConfig) -> dict[str, Any]:
    from callgraph.intent.data import customer_text

    labels = shared_labels(config, "intents")
    A, B, T = annotators(config)
    items = []
    for res in load_real_calls(config, list(config.evaluation.silver_labels["real_datasets"])):  # type: ignore[attr-defined]
        text = customer_text(res["transcript"]["turns"])
        votes = []
        for llm in (A, B):
            try:
                v = llm.complete_model(render_prompt("annotate_intent", config, labels=labels, customer_text=text[: int(config.evaluation.silver_labels["max_chars"])]), _IntentReply).intent  # type: ignore[attr-defined]
                votes.append(v if v in labels else None)
            except LLMError:
                votes.append(None)
        a, b = votes
        final = (
            a
            if a == b and a
            else tiebreak(
                T,
                config,
                "call reason",
                text[: int(config.evaluation.silver_labels["max_chars"])],
                a,
                b,
                labels,
            )
        )
        items.append(
            {
                "call_id": res["transcript"]["call_id"],
                "a": a,
                "b": b,
                "final": final,
                "tiebreak_used": a != b,
            }
        )
    return finish(config, "real_call_intent", items, {})


def annotate_extraction(config: AppConfig) -> dict[str, Any]:
    """Verify extracted fields on a random sample of processed calls (silver precision check)."""
    A, B, T = annotators(config)
    pool = [r for r in load_real_calls(config, list(config.evaluation.silver_labels["extraction_datasets"])) if r["modules"].get("extraction")]  # type: ignore[attr-defined]
    random.Random(int(config.split.seed)).shuffle(pool)  # type: ignore[attr-defined]
    items = []
    for res in pool[: int(config.evaluation.extraction["check_calls"])]:  # type: ignore[attr-defined]
        t = CallTranscript.model_validate(res["transcript"])
        ext = res["modules"]["extraction"]
        fields = {k: ext[k] for k in config.evaluation.silver_labels["extraction_fields"] if ext.get(k) not in (None, "", [])}  # type: ignore[attr-defined]
        got = []
        for llm in (A, B):
            try:
                got.append(llm.complete_model(render_prompt("annotate_extraction", config, transcript=format_transcript(t), fields=fields), _Verdicts).verdicts)  # type: ignore[attr-defined]
            except LLMError:
                got.append({})
        for f in fields:
            a, b = got[0].get(f), got[1].get(f)
            if a is None and b is None:
                continue
            final = (
                a
                if a == b
                else tiebreak(
                    T,
                    config,
                    f"is extracted field '{f}' supported by the transcript",
                    f"{format_transcript(t)}\nValue: {fields[f]}",
                    a,
                    b,
                    ["true", "false"],
                )
                == "true"
            )
            items.append(
                {
                    "call_id": t.call_id,
                    "field": f,
                    "value": fields[f],
                    "a": a,
                    "b": b,
                    "final": final,
                }
            )
    return finish(config, "extraction_check", items, {})


TASKS = {
    "emotion": annotate_emotion,
    "compliance": annotate_compliance,
    "intent": annotate_intent,
    "extraction": annotate_extraction,
}


def annotate(config: AppConfig, task: str) -> dict[str, Any]:
    return TASKS[task](config)
