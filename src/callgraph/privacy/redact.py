"""PII redaction with Presidio (entity types from config) before anything is stored."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from callgraph.config import AppConfig
from callgraph.schemas import CallTranscript


@lru_cache(maxsize=1)
def _engines(model: str) -> tuple[Any, Any]:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    nlp = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model}],
        }
    ).create_engine()
    return AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"]), AnonymizerEngine()


def redact_text(config: AppConfig, text: str) -> str:
    """Replace configured PII entities by placeholders."""
    p = config.privacy  # type: ignore[attr-defined]
    if not p.enabled or not text.strip():
        return text
    from presidio_anonymizer.entities import OperatorConfig

    analyzer, anonymizer = _engines(str(p.spacy_model))
    found = analyzer.analyze(
        text=text,
        language=str(p.language),
        entities=list(p.entities),
        score_threshold=float(p.score_threshold),
    )
    ops = {
        e: OperatorConfig("replace", {"new_value": str(p.placeholder_format).format(entity=e)})
        for e in p.entities
    }
    return anonymizer.anonymize(text=text, analyzer_results=found, operators=ops).text


def redact_call(config: AppConfig, call: CallTranscript) -> CallTranscript:
    """Return a copy of the call with redacted turn text (word lists dropped for safety)."""
    turns = [
        t.model_copy(update={"text": redact_text(config, t.text), "words": []}) for t in call.turns
    ]
    return call.model_copy(update={"turns": turns})
