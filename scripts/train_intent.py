"""Train intent models: ``python scripts/train_intent.py [baseline|transformer|all]``."""

from __future__ import annotations

import sys

from callgraph.config import get_config
from callgraph.intent import train
from callgraph.logging_setup import setup_logging


def main(what: str) -> None:
    """Run the requested training stage(s)."""
    setup_logging()
    cfg = get_config()
    if what in ("baseline", "all"):
        train.train_baseline(cfg)
    if what in ("transformer", "all"):
        train.train_transformer(cfg)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
