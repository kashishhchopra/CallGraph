"""M1: agent/customer role assignment (first-speaker heuristic + LLM greeting check)."""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel

from callgraph.config import AppConfig
from callgraph.llm.client import LLMClient, LLMError, render_prompt
from callgraph.schemas import Turn

AGENT, CUSTOMER = "agent", "customer"


class RoleGuess(BaseModel):
    """LLM reply for the role check."""

    agent_speaker: str
    confidence: float = 1.0


def first_speaker_roles(config: AppConfig, turns: list[Turn]) -> dict[str, str]:
    """Heuristic: the configured role goes to whoever speaks first; others get the opposite."""
    first = str(config.pipeline.roles["first_speaker_role"])  # type: ignore[attr-defined]
    other = CUSTOMER if first == AGENT else AGENT
    speakers = list(dict.fromkeys(t.speaker for t in turns))
    return {s: (first if i == 0 else other) for i, s in enumerate(speakers)}


def assign_roles(
    config: AppConfig, turns: list[Turn], llm: LLMClient | None = None
) -> tuple[list[Turn], dict[str, str]]:
    """Return turns with roles filled and the speaker->role map."""
    roles = first_speaker_roles(config, turns)
    rcfg = config.pipeline.roles  # type: ignore[attr-defined]
    if rcfg["llm_check_enabled"] and len(roles) > 1:
        head = turns[: int(rcfg["greeting_window_turns"])]
        try:
            client = llm or LLMClient(config)
            guess = client.complete_model(
                render_prompt("role_check", config, turns=head), RoleGuess  # type: ignore[arg-type]
            )
            if (
                guess.confidence >= float(config.thresholds.role_check["min_confidence"])
                and guess.agent_speaker in roles
            ):  # type: ignore[attr-defined]
                agent = guess.agent_speaker
                roles = {s: (AGENT if s == agent else CUSTOMER) for s in roles}
        except (LLMError, ValueError) as exc:
            logger.warning("Role LLM check failed, keeping heuristic: {}", exc)
    return [t.model_copy(update={"role": roles.get(t.speaker, CUSTOMER)}) for t in turns], roles
