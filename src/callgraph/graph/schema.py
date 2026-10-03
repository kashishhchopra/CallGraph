"""Graph schema loader: indexes, constraints, MERGE builders and LLM-facing schema text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from callgraph.config import AppConfig


@dataclass(frozen=True)
class NodeSpec:
    label: str
    key: str
    properties: list[str]


@dataclass(frozen=True)
class RelSpec:
    type: str
    src: str
    dst: str
    properties: list[str]


class GraphSchema:
    """Typed view of ``configs/graph_schema.yaml``."""

    def __init__(self, config: AppConfig) -> None:
        raw: dict[str, Any] = dict(config.graph_schema.model_extra or {})  # type: ignore[attr-defined]
        self.nodes = {
            k: NodeSpec(k, v["key"], list(v["properties"])) for k, v in raw["nodes"].items()
        }
        self.rels = {
            k: RelSpec(k, v["from"], v["to"], list(v["properties"]))
            for k, v in raw["relationships"].items()
        }
        self.vector: dict[str, Any] = dict(raw.get("vector", {}))

    def index_statements(self) -> list[str]:
        """Range index on every node key (needed for MERGE speed and constraints)."""
        return [f"CREATE INDEX FOR (n:{n.label}) ON (n.{n.key})" for n in self.nodes.values()]

    def vector_index_statement(self) -> str:
        v = self.vector
        return (
            f"CREATE VECTOR INDEX FOR (n:{v['node']}) ON (n.{v['property']}) "
            f"OPTIONS {{dimension:{v['dimension']}, similarityFunction:'{v['similarity']}'}}"
        )

    def merge_node(self, label: str, props: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Parameterised MERGE for a node; only schema properties are written."""
        spec = self.nodes[label]
        if spec.key not in props:
            raise KeyError(f"{label} requires key '{spec.key}'")
        body = {
            k: v
            for k, v in props.items()
            if k in spec.properties and k != spec.key and v is not None
        }
        sets = []
        for k in body:
            sets.append(
                f"n.{k} = vecf32($p_{k})"
                if label == self.vector.get("node") and k == self.vector.get("property")
                else f"n.{k} = $p_{k}"
            )
        q = f"MERGE (n:{label} {{{spec.key}: $key}})" + (" SET " + ", ".join(sets) if sets else "")
        return q, {"key": props[spec.key], **{f"p_{k}": v for k, v in body.items()}}

    def merge_rel(
        self, rel: str, src_key: Any, dst_key: Any, props: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Parameterised MERGE for a relationship between existing nodes (keys looked up from schema)."""
        r = self.rels[rel]
        body = {k: v for k, v in (props or {}).items() if k in r.properties and v is not None}
        sets = " SET " + ", ".join(f"r.{k} = $p_{k}" for k in body) if body else ""
        q = (
            f"MATCH (a:{r.src} {{{self.nodes[r.src].key}: $src}}), (b:{r.dst} {{{self.nodes[r.dst].key}: $dst}}) "
            f"MERGE (a)-[r:{rel}]->(b){sets}"
        )
        return q, {"src": src_key, "dst": dst_key, **{f"p_{k}": v for k, v in body.items()}}

    def schema_text(self) -> str:
        """Compact schema description for the text-to-Cypher prompt (built from config, not hand-written)."""
        lines = ["Nodes:"]
        for n in self.nodes.values():
            props = [p for p in n.properties if p != "embedding"]
            lines.append(f"  (:{n.label} {{{', '.join(props)}}})")
        lines.append("Relationships:")
        for r in self.rels.values():
            props = f" {{{', '.join(r.properties)}}}" if r.properties else ""
            lines.append(f"  (:{r.src})-[:{r.type}{props}]->(:{r.dst})")
        return "\n".join(lines)
