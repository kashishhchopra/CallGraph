"""M7: LLM structured extraction (validated, retried) as a registered module."""

from __future__ import annotations

from typing import Any

from callgraph.config import AppConfig
from callgraph.llm.client import LLMClient, render_prompt
from callgraph.registry import MODULES, AnalysisModule
from callgraph.schemas import CallTranscript, ExtractionOutput


def format_transcript(call: CallTranscript, timestamps: bool = False) -> str:
    """Render turns as 'ROLE: text' lines (optionally with [start] seconds)."""
    return "\n".join(
        (f"[{t.start:.0f}] " if timestamps else "") + f"{t.role.upper()}: {t.text}"
        for t in call.turns
    )


@MODULES.register("extraction")
class ExtractionModule(AnalysisModule):
    """Registered M7 plugin (extraction part; graph writing lives in ``graph_writer``)."""

    def run(self, call: CallTranscript, config: AppConfig, **context: Any) -> ExtractionOutput:
        llm: LLMClient = context.get("llm") or LLMClient(config)
        prompt = render_prompt("extraction", config, transcript=format_transcript(call), outcomes=list(config.models.extraction["outcomes"]))  # type: ignore[attr-defined]
        res = llm.complete_model(prompt, make_schema(config), max_retries=int(config.models.extraction["max_retries"]))  # type: ignore[attr-defined]
        return ExtractionOutput(call_id=call.call_id, **res.model_dump())  # type: ignore[attr-defined]


from pydantic import BaseModel, Field, create_model, field_validator  # noqa: E402


def make_schema(config: AppConfig) -> type[BaseModel]:
    """Validation schema for the LLM reply; allowed outcomes come from config."""
    outcomes = [str(o) for o in config.models.extraction["outcomes"]]  # type: ignore[attr-defined]

    def _outcome(cls: Any, v: str) -> str:
        v = str(v).strip().lower()
        if v not in outcomes:
            raise ValueError(f"outcome must be one of {outcomes}")
        return v

    def _listify(cls: Any, v: Any) -> Any:
        return [v] if isinstance(v, str) else (v or [])

    return create_model(
        "ExtractionReply",
        issue_type=(str, ...),
        product=(str | None, None),
        problem_description=(str, ""),
        agent_actions=(list[str], Field(default_factory=list)),
        resolution=(str, ""),
        outcome=(str, outcomes[-1]),
        follow_ups=(list[str], Field(default_factory=list)),
        __validators__={
            "o": field_validator("outcome")(classmethod(_outcome)),
            "l": field_validator("agent_actions", "follow_ups", mode="before")(
                classmethod(_listify)
            ),
        },
    )
