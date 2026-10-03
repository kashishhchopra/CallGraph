"""Configurable logging."""

from __future__ import annotations

import sys

from loguru import logger

from callgraph.config import get_config


def setup_logging() -> None:
    """Configure loguru from settings.yaml."""
    cfg = get_config().logging
    logger.remove()
    logger.add(sys.stderr, level=cfg.level, format=cfg.format)
