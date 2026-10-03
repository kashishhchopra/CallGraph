"""Plugin registries for datasets and analysis modules.

Adding a dataset or module = add a config entry + a class decorated with
``@DATASETS.register("name")`` / ``@MODULES.register("name")``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from callgraph.config import AppConfig
from callgraph.schemas import CallTranscript, ModuleOutput

T = TypeVar("T")


class Registry(Generic[T]):
    """Name -> class mapping with decorator registration."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            cls.name = name  # type: ignore[attr-defined]
            self._items[name] = cls
            return cls

        return deco

    def get(self, name: str) -> type[T]:
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Registered: {sorted(self._items)}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)


class AnalysisModule(ABC):
    """Common module interface: ``run(call, config) -> ModuleOutput``."""

    name: str = ""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @abstractmethod
    def run(self, call: CallTranscript, config: AppConfig, **context: Any) -> ModuleOutput:
        """Analyse one call."""

    def evaluate(self, config: AppConfig) -> dict[str, Any]:
        """Evaluation entry point (see ``eval/``). Override per module."""
        raise NotImplementedError(f"{self.name} has no evaluate()")

    def enabled(self, config: AppConfig) -> bool:
        """Read the enable flag from pipeline.yaml."""
        entry = getattr(config.pipeline, "modules", {}).get(self.name, {})
        return bool(entry.get("enabled", False))


class DatasetLoader(ABC):
    """Common dataset interface: download -> build manifest."""

    name: str = ""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.spec: dict[str, Any] = config.datasets[self.name]

    @abstractmethod
    def download(self) -> None:
        """Fetch (a subset of) the dataset from its official source."""

    @abstractmethod
    def build_manifest(self) -> Any:
        """Write the manifest CSV and return its path."""


DATASETS: Registry[DatasetLoader] = Registry("dataset")
MODULES: Registry[AnalysisModule] = Registry("module")
