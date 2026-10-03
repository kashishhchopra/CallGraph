"""Download helpers: disk-space guard, resumable HTTP, Hugging Face Hub."""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx
from loguru import logger

from callgraph.config import AppConfig


def free_gb(path: Path, bytes_per_gb: float) -> float:
    """Free disk space (GB) on the volume holding ``path``."""
    target = path if path.exists() else path.parent
    while not target.exists():
        target = target.parent
    return shutil.disk_usage(target).free / bytes_per_gb


def ensure_space(config: AppConfig, path: Path, needed_gb: float) -> None:
    """Raise if downloading ``needed_gb`` would leave less than the configured minimum free."""
    bpg = float(config.downloads.bytes_per_gb)  # type: ignore[attr-defined]
    remaining = free_gb(path, bpg) - needed_gb
    if remaining < float(config.downloads.min_free_gb):  # type: ignore[attr-defined]
        raise OSError(
            f"Not enough disk: {free_gb(path, bpg):.1f} GB free, need {needed_gb:.2f} GB "
            f"and must keep {config.downloads.min_free_gb} GB free. Reduce subset_size in datasets.yaml."  # type: ignore[attr-defined]
        )


def http_download(config: AppConfig, url: str, dest: Path, expected_gb: float = 0.0) -> Path:
    """Stream ``url`` to ``dest`` (skips if it exists)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dl = config.downloads  # type: ignore[attr-defined]
    dest.parent.mkdir(parents=True, exist_ok=True)
    ensure_space(config, dest, expected_gb)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": str(dl.user_agent)}
    logger.info("Downloading {}", url)
    with httpx.stream(
        "GET", url, headers=headers, follow_redirects=True, timeout=float(dl.http_timeout_s)
    ) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(int(dl.chunk_bytes)):
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def hf_download(
    config: AppConfig, repo_id: str, filename: str, dest_dir: Path, repo_type: str = "dataset"
) -> Path:
    """Download one file from the Hugging Face Hub into ``dest_dir``."""
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    token = config.hf_token.get_secret_value() if config.hf_token else None
    local = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=str(dest_dir),
        token=token,
    )
    return Path(local)
