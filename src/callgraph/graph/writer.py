"""Write one processed call into FalkorDB (schema-driven, idempotent MERGE)."""

from __future__ import annotations

from typing import Any

import numpy as np

from callgraph.config import AppConfig
from callgraph.extraction.normalize import Normalizer
from callgraph.graph.client import GraphClient
from callgraph.graph.library import run_query
from callgraph.schemas import CallResult
from callgraph.utils.embeddings import embed


class GraphWriter:
    """Convert a ``CallResult`` into nodes and relationships using ``graph_schema.yaml``."""

    def __init__(self, config: AppConfig, client: GraphClient | None = None) -> None:
        self.config = config
        self.g = client or GraphClient(config)
        self.g.ensure_schema()
        self.norm = {k: Normalizer(config, k) for k in ("issue", "action", "product")}

    def _node(self, label: str, props: dict[str, Any]) -> Any:
        q, p = self.g.schema.merge_node(label, props)
        self.g.query(q, p)
        return props[self.g.schema.nodes[label].key]

    def _rel(self, rel: str, a: Any, b: Any, props: dict[str, Any] | None = None) -> None:
        q, p = self.g.schema.merge_rel(rel, a, b, props)
        self.g.query(q, p)

    def write(self, res: CallResult) -> None:
        """Upsert the call and everything derived from it."""
        t = res.transcript
        m = res.modules
        cid = t.call_id
        run_query(self.g, "clear_call", call_id=cid)
        emo, intent = m.get("emotion", {}), m.get("intent", {})
        comp, summ, ext = m.get("compliance", {}), m.get("summary", {}), m.get("extraction", {})
        self._node(
            "Call",
            {
                "call_id": cid,
                "dataset": t.dataset,
                "duration_s": t.duration_s,
                "date": t.metadata.get("date"),
                "language": t.language,
                "wait_time_s": t.metadata.get("wait_time_s"),
                "frustration_score": emo.get("frustration_score"),
                "compliance_score": comp.get("score"),
                "summary": summ.get("summary"),
            },
        )
        spk_ids: dict[str, str] = {}
        for turn in t.turns:
            if turn.speaker not in spk_ids:
                spk_ids[turn.speaker] = self._node(
                    "Speaker", {"speaker_id": f"{cid}:{turn.speaker}", "role": turn.role}
                )
                self._rel("HAS_PARTICIPANT", cid, spk_ids[turn.speaker])
        emo_by_turn = {e["turn_index"]: e for e in emo.get("turns", [])}
        prev = None
        for i, turn in enumerate(t.turns):
            e = emo_by_turn.get(i, {})
            tid = self._node(
                "Turn",
                {
                    "turn_id": f"{cid}:{i}",
                    "index": i,
                    "start": turn.start,
                    "end": turn.end,
                    "text": turn.text,
                    "emotion": e.get("emotion"),
                    "emotion_conf": e.get("intensity"),
                },
            )
            self._rel("HAS_TURN", cid, tid)
            self._rel("SPOKEN_BY", tid, spk_ids[turn.speaker])
            if prev is not None:
                pt = t.turns[i - 1]
                self._rel(
                    "NEXT",
                    prev,
                    tid,
                    {
                        "gap_s": turn.start - pt.end,
                        "interrupted": bool(turn.start < pt.end and turn.speaker != pt.speaker),
                    },
                )
            prev = tid
        if intent:
            self._node("Intent", {"name": intent["top_intent"]})
            conf = next(
                (k["prob"] for k in intent["top_k"] if k["name"] == intent["top_intent"]), None
            )
            self._rel(
                "HAS_INTENT",
                cid,
                intent["top_intent"],
                {"confidence": conf, "detected_at_s": intent.get("detected_at_s")},
            )
        issue = None
        if ext:
            issue = self.norm["issue"].canonical(ext["issue_type"])
            vec = embed(self.config, [issue])[0].astype(np.float32).tolist()
            self._node(
                "Issue",
                {"name": issue, "description": ext.get("problem_description"), "embedding": vec},
            )
            self._rel("HAS_ISSUE", cid, issue)
            if ext.get("product"):
                prod = self.norm["product"].canonical(ext["product"])
                self._node("Product", {"name": prod})
                self._rel("ABOUT", issue, prod)
            for order, act in enumerate(ext.get("agent_actions", [])):
                a = self.norm["action"].canonical(act)
                if a:
                    self._node("Action", {"name": a})
                    self._rel("TOOK_ACTION", cid, a, {"order": order})
            self._node("Outcome", {"type": ext["outcome"]})
            self._rel("ENDED_WITH", cid, ext["outcome"])
        if emo.get("start_emotion"):
            for rel, key, inten in (
                ("STARTED_WITH", "start_emotion", "start_intensity"),
                ("ENDED_IN", "end_emotion", "end_intensity"),
            ):
                self._node("Emotion", {"name": emo[key]})
                self._rel(rel, cid, emo[key], {"intensity": emo.get(inten)})
        for item in comp.get("items", []):
            ck = self._node(
                "ComplianceCheck",
                {
                    "check_id": f"{cid}:{item['rubric_item']}",
                    "rubric_item": item["rubric_item"],
                    "passed": item["passed"],
                    "evidence": item.get("evidence", ""),
                },
            )
            self._rel("HAS_CHECK", cid, ck)

    def refresh_resolution_counts(self) -> None:
        """RESOLVED_BY(count): number of resolved calls where the issue's call took the action."""
        run_query(self.g, "refresh_resolution_counts", outcome=str(self.config.models.extraction["outcomes"][0]))  # type: ignore[attr-defined]
