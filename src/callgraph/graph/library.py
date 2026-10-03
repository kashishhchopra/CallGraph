"""Parameterised Cypher query library (``queries/*.cypher``): values are always passed as parameters."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from callgraph.graph.client import GraphClient

QUERY_DIR = Path(__file__).parent / "queries"
PARAMS_RE = re.compile(r"params:\s*(.*)")


def load_query(name: str) -> tuple[str, list[str], str]:
    """Return (cypher, parameter names, description) for ``queries/<name>.cypher``."""
    text = (QUERY_DIR / f"{name}.cypher").read_text()
    head, *body = text.splitlines()
    m = PARAMS_RE.search(head)
    params = [p.strip() for p in m.group(1).split(",")] if m else []
    return "\n".join(body), params, head.lstrip("/ ").split(" params:")[0]


def list_queries() -> list[str]:
    return sorted(p.stem for p in QUERY_DIR.glob("*.cypher"))


def run_query(client: GraphClient, name: str, **params: Any) -> list[dict[str, Any]]:
    """Execute a library query; every declared parameter must be supplied."""
    cypher, declared, _ = load_query(name)
    missing = [p for p in declared if p not in params]
    if missing:
        raise KeyError(f"Query '{name}' missing params: {missing}")
    return client.query(cypher, {k: params[k] for k in declared})
