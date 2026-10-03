"""Download datasets and/or build manifests: ``python scripts/dataset_cli.py download|manifests [names]``."""

from __future__ import annotations

import sys

from loguru import logger

import callgraph.data.loaders  # noqa: F401  (registers loaders)
from callgraph.config import get_config
from callgraph.logging_setup import setup_logging
from callgraph.registry import DATASETS


def main(argv: list[str]) -> int:
    """Run ``download`` or ``manifests`` for the named datasets (default: all registered)."""
    setup_logging()
    cfg = get_config()
    action, names = argv[0], argv[1:] or DATASETS.names()
    failed = []
    for name in names:
        loader = DATASETS.get(name)(cfg)
        try:
            if action == "download":
                loader.download()
            else:
                loader.build_manifest()
            logger.info("{} {}: ok", action, name)
        except Exception as exc:  # keep going: other datasets must not wait on one failure
            logger.error("{} {} failed: {}", action, name, exc)
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
