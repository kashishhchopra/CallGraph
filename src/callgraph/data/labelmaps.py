"""Label harmonisation: source labels -> shared sets. Fails loudly on unmapped labels."""

from __future__ import annotations

from typing import Any

from callgraph.config import AppConfig


class UnmappedLabelError(KeyError):
    """A source label has no entry in the label-map YAML."""


def _section(config: AppConfig, kind: str) -> dict[str, Any]:
    return dict(getattr(config, kind).model_extra or {})


def shared_labels(config: AppConfig, kind: str) -> list[str]:
    """Shared label set for ``kind`` in {'intents', 'emotions'}."""
    return list(_section(config, kind)["shared_labels"])


def map_label(
    config: AppConfig, kind: str, dataset: str, source: str, flow: str | None = None
) -> str:
    """Map one source label to the shared set; raise if unmapped or outside the shared set."""
    ds = _section(config, kind)["datasets"].get(dataset)
    if ds is None:
        raise UnmappedLabelError(f"No {kind} mapping for dataset '{dataset}'")
    if "flows" in ds:  # hierarchical (flow, subflow) mapping
        table = ds["flows"].get(flow or "")
        if table is None:
            raise UnmappedLabelError(f"{dataset}: unmapped flow '{flow}'")
        target = table.get(source, table.get("*"))
    else:
        target = ds.get(source)
    if target is None:
        raise UnmappedLabelError(f"{dataset}: unmapped {kind} label '{flow}/{source}'")
    if target not in shared_labels(config, kind):
        raise UnmappedLabelError(f"{dataset}: '{source}' maps to '{target}', not in shared set")
    return str(target)
