"""Evaluation orchestrator: ``python -m eval.run [runner ...]`` (runners and experiment flags from evaluation.yaml)."""

from __future__ import annotations

import importlib
import sys
import time
import traceback

from loguru import logger

from callgraph.config import get_config
from callgraph.logging_setup import setup_logging
from eval.common import run_dir, save_result

EXPERIMENT_RUNNERS = {"dynamics_vs_outcome": "eval.exp6"}


def main(names: list[str]) -> None:
    setup_logging()
    cfg = get_config()
    ev = cfg.evaluation  # type: ignore[attr-defined]
    chosen = names or list(ev.runners)
    modules = {n: EXPERIMENT_RUNNERS.get(n, f"eval.{n}") for n in chosen}
    if not names and ev.experiments["dynamics_vs_outcome"]:
        modules["dynamics_vs_outcome"] = EXPERIMENT_RUNNERS["dynamics_vs_outcome"]
    logger.info("Run dir: {}", run_dir(cfg))
    status = {}
    for name, mod in modules.items():
        t0 = time.time()
        try:
            importlib.import_module(mod).run(cfg)
            status[name] = f"ok ({time.time() - t0:.0f}s)"
        except (
            Exception
        ) as exc:  # one failing runner must not block the rest; the error is recorded
            logger.error("{} failed: {}", name, exc)
            status[name] = f"FAILED: {exc}"
            (run_dir(cfg) / f"{name}.error.txt").write_text(traceback.format_exc())
        logger.info("{} -> {}", name, status[name])
    save_result(cfg, f"run_status_{int(time.time())}", status)


if __name__ == "__main__":
    main(sys.argv[1:])
