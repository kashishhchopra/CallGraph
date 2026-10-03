"""Tune late-fusion weights on MELD validation: ``python scripts/tune_fusion.py``."""

from __future__ import annotations

from loguru import logger

from callgraph.config import get_config
from callgraph.emotion.tune import tune_fusion
from callgraph.logging_setup import setup_logging

if __name__ == "__main__":
    setup_logging()
    logger.info("Fusion weights: {}", tune_fusion(get_config()))
