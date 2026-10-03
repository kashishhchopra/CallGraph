"""M5: rubric-based compliance audit (LLM) with keyword-rule baseline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from callgraph.config import AppConfig
from callgraph.extraction.extract import format_transcript
from callgraph.llm.client import LLMClient, render_prompt
from callgraph.registry import MODULES, AnalysisModule
from callgraph.schemas import CallTranscript, ComplianceItemResult, ComplianceOutput


class _Reply(BaseModel):
    items: list[ComplianceItemResult]


def rubric_items(config: AppConfig) -> list[dict[str, Any]]:
    return list(config.rubric.items)  # type: ignore[attr-defined]


def score_items(
    config: AppConfig, call_id: str, items: list[ComplianceItemResult]
) -> ComplianceOutput:
    """Score = pass_points per passed item (scale from rubric config); missed steps listed."""
    sc = config.rubric.scoring  # type: ignore[attr-defined]
    passed = sum(1 for i in items if i.passed)
    return ComplianceOutput(
        call_id=call_id,
        items=items,
        score=min(float(sc["scale"]), passed * float(sc["pass_points"])),
        missed_steps=[i.rubric_item for i in items if not i.passed],
    )


def _applicable(item: dict[str, Any], upset: bool) -> bool:
    return item.get("applies_when", "always") == "always" or upset


def audit_llm(
    config: AppConfig,
    call: CallTranscript,
    intent: str,
    upset: bool,
    llm: LLMClient | None = None,
    guidelines: str | None = None,
) -> ComplianceOutput:
    """LLM audit: every rubric item judged; inapplicable items auto-pass."""
    client = llm or LLMClient(config)
    rub = rubric_items(config)
    prompt = render_prompt(
        "compliance",
        config,
        intent=intent,
        upset=upset,
        rubric=rub,
        guidelines=guidelines,
        transcript=format_transcript(call, timestamps=True),
    )
    reply = client.complete_model(prompt, _Reply, max_retries=int(config.thresholds.compliance["max_retries"]))  # type: ignore[attr-defined]
    by_id = {i.rubric_item: i for i in reply.items}  # type: ignore[attr-defined]
    na_text = str(config.rubric.scoring["na_evidence"])  # type: ignore[attr-defined]
    items = []
    for it in rub:
        r = by_id.get(it["id"], ComplianceItemResult(rubric_item=it["id"], passed=False))
        if not _applicable(it, upset):
            r = ComplianceItemResult(rubric_item=it["id"], passed=True, evidence=na_text)
        items.append(r)
    return score_items(config, call.call_id, items)


def audit_keywords(config: AppConfig, call: CallTranscript, upset: bool) -> ComplianceOutput:
    """Keyword baseline: an item passes if any configured keyword appears in agent speech."""
    kw = config.rubric.keyword_baseline  # type: ignore[attr-defined]
    agent = [t for t in call.turns if t.role == "agent"]
    na_text = str(config.rubric.scoring["na_evidence"])  # type: ignore[attr-defined]
    items = []
    for it in rubric_items(config):
        if not _applicable(it, upset):
            items.append(ComplianceItemResult(rubric_item=it["id"], passed=True, evidence=na_text))
            continue
        hit = next(((t, k) for t in agent for k in kw[it["id"]] if k in t.text.lower()), None)
        items.append(
            ComplianceItemResult(
                rubric_item=it["id"],
                passed=hit is not None,
                evidence=hit[0].text if hit else "",
                timestamp=hit[0].start if hit else None,
            )
        )
    return score_items(config, call.call_id, items)


@MODULES.register("compliance")
class ComplianceModule(AnalysisModule):
    """Registered M5 plugin. Context: ``intent`` (IntentOutput), ``emotion`` (EmotionOutput), ``llm``."""

    def run(self, call: CallTranscript, config: AppConfig, **context: Any) -> ComplianceOutput:
        intent = context["intent"].top_intent if context.get("intent") else "unknown"
        neg = set(config.thresholds.emotion["negative_labels"])  # type: ignore[attr-defined]
        emo = context.get("emotion")
        upset = bool(emo and any(t.emotion in neg for t in emo.turns))
        return audit_llm(config, call, intent, upset, context.get("llm"))
