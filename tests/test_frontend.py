"""The web UI is served by the API and contains no hard-coded label sets (they come from /config/public)."""

from __future__ import annotations

import re
from pathlib import Path

from api.main import app
from fastapi.testclient import TestClient

from callgraph.config import get_config

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
client = TestClient(app)


def test_ui_is_served_and_assets_resolve() -> None:
    html = client.get("/")
    assert html.status_code == 200 and "<title>CallGraph</title>" in html.text
    for asset in re.findall(r'(?:src|href)="(/[^"]+)"', html.text):
        assert client.get(asset).status_code == 200, asset
    for js in (FRONTEND / "js").glob("*.js"):
        for imp in re.findall(r'from "\./([^"]+)"', js.read_text()):
            assert (FRONTEND / "js" / imp).exists(), f"{js.name} imports missing {imp}"


def test_public_config_has_everything_the_ui_reads() -> None:
    cfg = client.get("/config/public").json()
    for key in (
        "nav",
        "workspace",
        "user",
        "analytics_cards",
        "node_colors",
        "emotion_colors",
        "intents",
        "outcomes",
        "strategies",
        "example_questions",
        "pipeline_stages",
        "default_strategy",
        "compliance_scale",
    ):
        assert key in cfg, key
    assert cfg["default_strategy"] in cfg["strategies"]


def test_frontend_hardcodes_no_label_sets() -> None:
    conf = get_config()
    labels = set(conf.intents.model_extra["shared_labels"]) | set(conf.emotions.model_extra["shared_labels"])  # type: ignore[index]
    labels |= set(conf.models.extraction["outcomes"])  # type: ignore[attr-defined]
    labels |= {i["id"] for i in conf.rubric.items}  # type: ignore[attr-defined]
    source = "\n".join(p.read_text() for p in (FRONTEND / "js").glob("*.js"))
    quoted = set(re.findall(r"""["'`]([a-z_]+)["'`]""", source))
    assert not (quoted & labels), quoted & labels


def test_overview_and_calls_endpoints() -> None:
    assert "kpis" in client.get("/overview").json()
    assert "items" in client.get("/calls", params={"q": "zzz-no-such-call"}).json()
