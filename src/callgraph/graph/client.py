"""FalkorDB client wrapper."""

from __future__ import annotations

from typing import Any

from falkordb import FalkorDB
from loguru import logger

from callgraph.config import AppConfig, get_config
from callgraph.graph.schema import GraphSchema


class GraphClient:
    """Connect to FalkorDB and run parameterised Cypher."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        f = self.config.falkordb
        self.db = FalkorDB(host=f.host, port=int(f.port))
        self.graph = self.db.select_graph(f.graph_name)
        self.schema = GraphSchema(self.config)

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute and return rows as dicts keyed by column header."""
        res = self.graph.query(cypher, params or {})
        cols = [h[1] if isinstance(h, (list, tuple)) else h for h in res.header]
        return [dict(zip(cols, row, strict=False)) for row in res.result_set]

    def ro_query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Read-only execution (used for LLM-generated Cypher)."""
        res = self.graph.ro_query(cypher, params or {})
        cols = [h[1] if isinstance(h, (list, tuple)) else h for h in res.header]
        return [dict(zip(cols, row, strict=False)) for row in res.result_set]

    def ensure_schema(self) -> None:
        """Create indexes (idempotent: existing-index errors are ignored)."""
        for stmt in [*self.schema.index_statements(), self.schema.vector_index_statement()]:
            try:
                self.graph.query(stmt)
            except Exception as exc:  # index already exists
                logger.debug("index stmt skipped ({}): {}", stmt, exc)

    def healthy(self) -> bool:
        try:
            return bool(self.db.connection.ping())
        except Exception:
            return False

    def clear(self) -> None:
        """Delete the whole graph (used by tests/eval rebuilds)."""
        try:
            self.graph.delete()
        except Exception as exc:
            logger.debug("graph delete: {}", exc)
        self.graph = self.db.select_graph(self.config.falkordb.graph_name)
