"""M4: rule-based conversation dynamics from timed turns."""

from __future__ import annotations

import operator
from typing import Any

from callgraph.config import AppConfig
from callgraph.registry import MODULES, AnalysisModule
from callgraph.schemas import CallTranscript, DynamicsOutput, Turn

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}
SECONDS_PER_MINUTE = 60.0


def overlaps(turns: list[Turn], min_overlap_s: float) -> list[dict[str, Any]]:
    """Interruptions: a different speaker starts before the previous turn ends by > ``min_overlap_s``."""
    out = []
    for prev, cur in zip(turns, turns[1:], strict=False):
        ov = prev.end - cur.start
        if prev.speaker != cur.speaker and ov > 0:
            out.append(
                {
                    "start": cur.start,
                    "end": min(prev.end, cur.end),
                    "overlap_s": ov,
                    "interrupter": cur.speaker,
                    "interruption": ov > min_overlap_s,
                }
            )
    return out


def gaps(turns: list[Turn]) -> list[dict[str, float]]:
    """Silences between consecutive turns (any speakers)."""
    return [
        {"start": a.end, "end": b.start, "gap_s": b.start - a.end}
        for a, b in zip(turns, turns[1:], strict=False)
        if b.start > a.end
    ]


def compute(
    config: AppConfig, call: CallTranscript, wait_time_s: float | None = None
) -> DynamicsOutput:
    """Compute the dynamics vector and flags for one call."""
    th = config.thresholds.dynamics  # type: ignore[attr-defined]
    turns = call.turns
    talk = {"agent": 0.0, "customer": 0.0}
    words = {"agent": 0, "customer": 0}
    for t in turns:
        if t.role in talk:
            talk[t.role] += max(0.0, t.end - t.start)
            words[t.role] += len(t.text.split())
    total = sum(talk.values()) or 1.0
    ov = overlaps(turns, float(th["interruption_overlap_s"]))
    gp = gaps(turns)
    long_gaps = [g for g in gp if g["gap_s"] > float(th["long_silence_s"])]
    holds = [g for g in gp if g["gap_s"] > float(th["hold_period_s"])]
    feats: dict[str, float] = {
        "agent_talk_ratio": talk["agent"] / total,
        "customer_talk_ratio": talk["customer"] / total,
        "interruptions": float(sum(1 for o in ov if o["interruption"])),
        "overlap_time_s": float(sum(o["overlap_s"] for o in ov)),
        "long_silences": float(len(long_gaps)),
        "dead_air_total_s": float(sum(g["gap_s"] for g in long_gaps)),
        "agent_wpm": words["agent"] / max(talk["agent"], 1e-9) * SECONDS_PER_MINUTE,
        "customer_wpm": words["customer"] / max(talk["customer"], 1e-9) * SECONDS_PER_MINUTE,
        "num_turns": float(len(turns)),
        "avg_turn_len_s": float(sum(t.end - t.start for t in turns) / max(len(turns), 1)),
        "hold_periods": float(len(holds)),
    }
    if wait_time_s is not None:
        feats["wait_time_s"] = float(wait_time_s)
    flags = [
        name
        for name, rule in th["flags"].items()
        if rule["metric"] in feats and OPS[rule["op"]](feats[rule["metric"]], float(rule["value"]))
    ]
    return DynamicsOutput(call_id=call.call_id, features=feats, flags=flags, gaps=long_gaps)


@MODULES.register("dynamics")
class DynamicsModule(AnalysisModule):
    """Registered M4 plugin."""

    def run(self, call: CallTranscript, config: AppConfig, **context: Any) -> DynamicsOutput:
        return compute(config, call, call.metadata.get("wait_time_s"))
