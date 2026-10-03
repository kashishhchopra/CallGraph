"""Build the GraphRAG question set from the populated graph: ``python scripts/build_questions.py``."""

from __future__ import annotations

from eval.questions import build

from callgraph.config import get_config
from callgraph.logging_setup import setup_logging

if __name__ == "__main__":
    setup_logging()
    build(get_config())
