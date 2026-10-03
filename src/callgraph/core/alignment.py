"""M1: word-to-speaker alignment and turn building."""

from __future__ import annotations

from callgraph.config import AppConfig
from callgraph.core.diarization import SpeakerSegment
from callgraph.schemas import Turn, Word


def speaker_for(word: Word, segs: list[SpeakerSegment]) -> str | None:
    """Speaker with maximal overlap with the word; nearest segment if none overlaps."""
    best, best_ov = None, 0.0
    for s in segs:
        ov = min(word.end, s.end) - max(word.start, s.start)
        if ov > best_ov:
            best, best_ov = s.speaker, ov
    if best is not None or not segs:
        return best
    mid = (word.start + word.end) / 2
    return min(segs, key=lambda s: min(abs(mid - s.start), abs(mid - s.end))).speaker


def build_turns(config: AppConfig, words: list[Word], segs: list[SpeakerSegment]) -> list[Turn]:
    """Group aligned words into speaker turns (role filled in later)."""
    a = config.pipeline.alignment  # type: ignore[attr-defined]
    gap, min_words = float(a["turn_merge_gap_s"]), int(a["min_turn_words"])
    turns: list[Turn] = []
    cur: list[Word] = []
    cur_spk: str | None = None

    def flush() -> None:
        if len(cur) >= min_words and cur_spk is not None:
            probs = [w.prob for w in cur if w.prob is not None]
            turns.append(
                Turn(
                    speaker=cur_spk,
                    role="unknown",
                    start=cur[0].start,
                    end=cur[-1].end,
                    text=" ".join(w.text for w in cur),
                    confidence=sum(probs) / len(probs) if probs else None,
                    words=list(cur),
                )
            )

    for w in sorted(words, key=lambda x: x.start):
        spk = speaker_for(w, segs) or cur_spk
        w = w.model_copy(update={"speaker": spk})
        if cur and (spk != cur_spk or w.start - cur[-1].end > gap):
            flush()
            cur = []
        cur_spk = spk
        cur.append(w)
    flush()
    return turns
