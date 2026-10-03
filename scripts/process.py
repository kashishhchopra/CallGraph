"""Process calls: ``python scripts/process.py <dataset> [--limit N] [--core-only] [--force] [--no-graph]``."""

from __future__ import annotations

import argparse
import time

from loguru import logger

from callgraph.config import get_config
from callgraph.core.m1 import run_core, save_transcript
from callgraph.data.manifest import read_manifest
from callgraph.logging_setup import setup_logging
from callgraph.pipeline import Pipeline


def main() -> None:
    """Run M1 (and optionally the whole pipeline) for the audio rows of a dataset manifest."""
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--core-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()
    setup_logging()
    cfg = get_config()
    man = read_manifest(cfg, args.dataset)
    man = man[man["audio_path"].notna()].head(args.limit)
    pipe = None if args.core_only else Pipeline(cfg, write_graph=not args.no_graph)
    t0 = time.time()
    for _, r in man.iterrows():
        meta = {
            k: v
            for k, v in (r["labels"] or {}).items()
            if k in ("wait_time_s", "date", "category", "company", "language_hint", "domain")
        }
        audio = cfg.root / r["audio_path"]
        if pipe is None:
            save_transcript(cfg, run_core(cfg, audio, str(r["id"]), args.dataset, meta))
        else:
            pipe.process_audio(audio, str(r["id"]), args.dataset, meta, force=args.force)
        logger.info("[{:.0f}s] done {}", time.time() - t0, r["id"])
    if pipe is not None and not args.no_graph:
        pipe.writer.refresh_resolution_counts()


if __name__ == "__main__":
    main()
