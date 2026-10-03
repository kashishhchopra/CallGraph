"""Regression checks on the container/runtime setup (the API container also serves the web UI)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_sets_pythonpath() -> None:
    assert "PYTHONPATH=/app" in (ROOT / "Dockerfile").read_text()


def test_compose_has_no_separate_dashboard_and_api_serves_ui() -> None:
    svc = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    assert "dashboard" not in svc and svc["api"]["image"]
    assert any(v.startswith("./data") for v in svc["api"]["volumes"])


def test_makefile_serve_sets_pythonpath() -> None:
    assert "PYTHONPATH=. $(PY) -m uvicorn api.main:app" in (ROOT / "Makefile").read_text()
