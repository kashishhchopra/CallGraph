"""Fine-tune the seq2seq summariser on DialogSum: ``python scripts/train_summary.py``."""

from __future__ import annotations

from callgraph.config import get_config
from callgraph.logging_setup import setup_logging
from callgraph.summarization.train import train

if __name__ == "__main__":
    setup_logging()
    train(get_config())
