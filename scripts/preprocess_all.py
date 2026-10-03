"""Write preprocessed 16 kHz mono copies (and 8 kHz telephone-simulated copies for emotion data)."""

from __future__ import annotations

from loguru import logger

from callgraph.config import get_config
from callgraph.core.preprocess import preprocess_file
from callgraph.data.manifest import read_manifest
from callgraph.logging_setup import setup_logging

if __name__ == "__main__":
    setup_logging()
    cfg = get_config()
    for ds, spec in cfg.datasets.items():
        try:
            man = read_manifest(cfg, ds)
        except FileNotFoundError:
            continue
        for _, r in man[man["audio_path"].notna()].iterrows():
            phone = "emotion" in spec.get("used_by_modules", [])
            dst = cfg.path("processed_dir") / "audio" / ds / f"{r['id']}.wav"
            if not dst.exists():
                preprocess_file(cfg, cfg.root / r["audio_path"], dst, trim=True, phone=False)
            if phone:
                pdst = dst.with_name(f"{r['id']}_8k.wav")
                if not pdst.exists():
                    preprocess_file(cfg, cfg.root / r["audio_path"], pdst, trim=False, phone=True)
        logger.info("preprocessed {}", ds)
