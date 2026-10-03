"""Train emotion models: ``python scripts/train_emotion.py [text|audio|all]``."""

from __future__ import annotations

import sys

from callgraph.config import get_config
from callgraph.emotion import train
from callgraph.logging_setup import setup_logging

if __name__ == "__main__":
    setup_logging()
    cfg, what = get_config(), (sys.argv[1] if len(sys.argv) > 1 else "all")
    if what in ("text", "all"):
        train.train_text(cfg)
    if what in ("audio", "all"):
        train.train_audio(cfg)
