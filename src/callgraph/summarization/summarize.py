"""M6: call summarisation with interchangeable backends (prompted LLM or fine-tuned seq2seq)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import BaseModel, field_validator

from callgraph.config import AppConfig
from callgraph.extraction.extract import format_transcript
from callgraph.llm.client import LLMClient, render_prompt
from callgraph.registry import MODULES, AnalysisModule
from callgraph.schemas import CallTranscript, SummaryOutput


class _Reply(BaseModel):
    issue: str
    agent_actions: str
    outcome: str
    follow_up_needed: bool
    summary: str

    @field_validator("issue", "agent_actions", "outcome", "summary", mode="before")
    @classmethod
    def _flatten(cls, v: Any) -> Any:
        return "; ".join(str(x) for x in v) if isinstance(v, list) else v


@lru_cache(maxsize=1)
def _seq2seq(path: str, dev: str) -> tuple[Any, Any]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    return (
        AutoTokenizer.from_pretrained(path),
        AutoModelForSeq2SeqLM.from_pretrained(path).to(dev).eval(),
    )


def summarize_text_seq2seq(config: AppConfig, text: str) -> str:
    """Generate a summary with the fine-tuned seq2seq model."""
    import torch

    from callgraph.utils.device import pick_device

    m = config.models.summarization  # type: ignore[attr-defined]
    dev = str(pick_device(str(m.get("device", "auto"))))
    tok, model = _seq2seq(str(config.root / m["output_dir"]), dev)
    enc = tok(text, truncation=True, max_length=int(m["max_input_length"]), return_tensors="pt").to(
        dev
    )
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=int(m["max_target_length"]), num_beams=int(m.get("num_beams", 2))
        )
    return tok.decode(out[0], skip_special_tokens=True)


def summarize_llm(
    config: AppConfig, call: CallTranscript, llm: LLMClient | None = None
) -> SummaryOutput:
    """Prompted LLM summary with the structured fields."""
    m = config.models.summarization  # type: ignore[attr-defined]
    prompt = render_prompt(
        "summarization",
        config,
        transcript=format_transcript(call),
        min_sentences=m["min_sentences"],
        max_sentences=m["max_sentences"],
    )
    r = (llm or LLMClient(config)).complete_model(prompt, _Reply)
    return SummaryOutput(call_id=call.call_id, **r.model_dump())  # type: ignore[attr-defined]


@MODULES.register("summary")
class SummaryModule(AnalysisModule):
    """Registered M6 plugin; backend chosen by ``models.summarization.backend``."""

    def run(self, call: CallTranscript, config: AppConfig, **context: Any) -> SummaryOutput:
        if config.models.summarization["backend"] == "seq2seq":  # type: ignore[attr-defined]
            llm_part = summarize_llm(config, call, context.get("llm"))
            text = summarize_text_seq2seq(config, format_transcript(call))
            return llm_part.model_copy(update={"summary": text})
        return summarize_llm(config, call, context.get("llm"))
