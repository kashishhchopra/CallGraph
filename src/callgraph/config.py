"""Typed, validated configuration loaded from ``configs/*.yaml`` and ``.env``.

Environment variables override YAML using ``__`` as nesting delimiter,
e.g. ``FALKORDB__HOST=db`` or ``LLM__MODEL=llama3.2:1b``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

ROOT_ENV_VAR = "CALLGRAPH_ROOT"
CONFIG_DIRNAME = "configs"


class Section(BaseModel):
    """Config section that tolerates extra keys (module-specific settings)."""

    model_config = ConfigDict(extra="allow")


class Paths(Section):
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    manifests_dir: Path
    labels_dir: Path
    cache_dir: Path
    models_dir: Path
    outputs_dir: Path
    prompts_dir: Path
    configs_dir: Path


class LoggingCfg(Section):
    level: str
    format: str


class FalkorCfg(Section):
    host: str
    port: int
    graph_name: str


class LLMCfg(Section):
    provider: str
    model: str
    base_url: str
    timeout_s: float
    max_retries: int
    temperature: float
    json_mode: bool
    cache_enabled: bool


class ApiCfg(Section):
    host: str
    port: int
    default_page_size: int
    max_page_size: int


class DashboardInfraCfg(Section):
    port: int
    api_base_url: str


class VectorCfg(Section):
    backend: str
    embedding_model: str
    chunk_size_turns: int
    chunk_overlap_turns: int


def find_root() -> Path:
    """Locate the project root (directory containing ``configs/``)."""
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        return Path(env).resolve()
    for start in (Path.cwd(), Path(__file__).resolve()):
        for parent in [start, *start.parents]:
            if (parent / CONFIG_DIRNAME / "settings.yaml").exists():
                return parent
    raise FileNotFoundError("Cannot locate project root with configs/settings.yaml")


def _read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_all(root: Path) -> dict[str, Any]:
    cfg = root / CONFIG_DIRNAME
    data: dict[str, Any] = dict(_read_yaml(cfg / "settings.yaml"))
    data["pipeline"] = _read_yaml(cfg / "pipeline.yaml")
    data["thresholds"] = _read_yaml(cfg / "thresholds.yaml")
    data["models"] = _read_yaml(cfg / "models.yaml")
    data["evaluation"] = _read_yaml(cfg / "evaluation.yaml")
    data["dashboard_ui"] = _read_yaml(cfg / "dashboard.yaml")
    data["graph_schema"] = _read_yaml(cfg / "graph_schema.yaml")
    data["datasets"] = _read_yaml(cfg / "datasets.yaml").get("datasets", {})
    data["intents"] = _read_yaml(cfg / "labels" / "intents.yaml")
    data["emotions"] = _read_yaml(cfg / "labels" / "emotions.yaml")
    data["rubric"] = _read_yaml(cfg / "rubrics" / "compliance.yaml")
    data["rag"] = _read_yaml(cfg / "rag.yaml")
    data["analytics"] = _read_yaml(cfg / "analytics.yaml")
    data["rag_questions"] = _read_yaml(cfg / "rag_questions.yaml")
    data["root"] = root
    return data


class AppConfig(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__", env_file=".env", extra="ignore", arbitrary_types_allowed=True
    )

    root: Path
    paths: Paths
    logging: LoggingCfg
    falkordb: FalkorCfg
    llm: LLMCfg
    api: ApiCfg
    dashboard: DashboardInfraCfg
    features: Section
    tracking: Section
    vector_store: VectorCfg
    downloads: Section
    privacy: Section
    rttm: Section
    manifest: Section
    split: Section
    pipeline: Section
    thresholds: Section
    models: Section
    evaluation: Section
    dashboard_ui: Section
    graph_schema: Section
    datasets: dict[str, dict[str, Any]]
    intents: Section
    emotions: Section
    rubric: Section
    rag: Section
    analytics: Section
    rag_questions: Section
    hf_token: SecretStr | None = None
    llm_api_key: SecretStr | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Environment beats ``.env`` beats YAML (init kwargs)."""
        return env_settings, dotenv_settings, init_settings

    def path(self, name: str) -> Path:
        """Absolute path for a configured path key, relative to project root."""
        p: Path = getattr(self.paths, name)
        return p if p.is_absolute() else self.root / p


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load (once) and return the validated configuration."""
    root = find_root()
    return AppConfig(**_load_all(root))


def reload_config() -> AppConfig:
    """Drop the cache and reload from disk."""
    get_config.cache_clear()
    return get_config()
