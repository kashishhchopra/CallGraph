"""Pydantic models for every JSON artefact produced by the pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Word(BaseModel):
    """A single ASR word with timing."""

    text: str
    start: float
    end: float
    prob: float | None = None
    speaker: str | None = None


class Turn(BaseModel):
    """A speaker turn."""

    speaker: str
    role: str
    start: float
    end: float
    text: str
    confidence: float | None = None
    words: list[Word] = Field(default_factory=list)


class CallTranscript(BaseModel):
    """Output of M1 and input to every analysis module."""

    call_id: str
    duration_s: float
    language: str = "en"
    language_prob: float | None = None
    non_english: bool = False
    dataset: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    turns: list[Turn]


class ModuleOutput(BaseModel):
    """Base class for every module output."""

    module: str
    call_id: str


class IntentScore(BaseModel):
    name: str
    prob: float


class IntentOutput(ModuleOutput):
    module: str = "intent"
    top_intent: str
    top_k: list[IntentScore]
    secondary: list[IntentScore] = Field(default_factory=list)
    detected_at_s: float | None = None
    by_cutoff: dict[str, str] = Field(default_factory=dict)


class TurnEmotion(BaseModel):
    turn_index: int
    emotion: str
    intensity: float
    probs: dict[str, float] = Field(default_factory=dict)


class EmotionOutput(ModuleOutput):
    module: str = "emotion"
    turns: list[TurnEmotion]
    start_emotion: str | None = None
    end_emotion: str | None = None
    start_intensity: float | None = None
    end_intensity: float | None = None
    peak_anger: float = 0.0
    shift: Literal["improved", "worsened", "stable"] = "stable"
    frustration_score: float = 0.0
    ends_negative: bool = False


class DynamicsOutput(ModuleOutput):
    module: str = "dynamics"
    features: dict[str, float]
    flags: list[str] = Field(default_factory=list)
    gaps: list[dict[str, Any]] = Field(default_factory=list)


class ComplianceItemResult(BaseModel):
    rubric_item: str
    passed: bool
    evidence: str | None = ""
    timestamp: float | None = None


class ComplianceOutput(ModuleOutput):
    module: str = "compliance"
    items: list[ComplianceItemResult]
    score: float
    missed_steps: list[str] = Field(default_factory=list)


class SummaryOutput(ModuleOutput):
    module: str = "summary"
    issue: str
    agent_actions: str
    outcome: str
    follow_up_needed: bool
    summary: str


class ExtractionOutput(ModuleOutput):
    module: str = "extraction"
    issue_type: str
    product: str | None = None
    problem_description: str = ""
    agent_actions: list[str] = Field(default_factory=list)
    resolution: str = ""
    outcome: str = "unresolved"
    follow_ups: list[str] = Field(default_factory=list)


class CallResult(BaseModel):
    """Everything known about one processed call."""

    transcript: CallTranscript
    modules: dict[str, dict[str, Any]] = Field(default_factory=dict)
    status: dict[str, str] = Field(default_factory=dict)
