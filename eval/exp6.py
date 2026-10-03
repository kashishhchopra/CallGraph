"""Experiment 6: correlation between dynamics features and negative call endings (needs processed audio calls)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from callgraph.config import AppConfig
from eval.common import save_result


def run(config: AppConfig) -> dict[str, Any]:
    rows = []
    for p in (config.path("processed_dir") / "calls").glob("*/result.json"):
        m = json.loads(p.read_text())["modules"]
        if m.get("dynamics") and m.get("emotion", {}).get("turns"):
            rows.append(
                {
                    **m["dynamics"]["features"],
                    "ends_negative": float(m["emotion"]["ends_negative"]),
                    "frustration_score": m["emotion"]["frustration_score"],
                }
            )
    df = pd.DataFrame(rows)
    if len(df) < 3:  # noqa: PLR2004
        return {"note": f"only {len(df)} calls with dynamics+emotion"}
    corr = df.corr(numeric_only=True)[["ends_negative", "frustration_score"]].drop(
        ["ends_negative", "frustration_score"], errors="ignore"
    )
    table = corr.reset_index().rename(columns={"index": "feature"})
    save_result(
        config,
        "exp6_dynamics_vs_outcome",
        {
            "n_calls": len(df),
            "rows": table.to_dict("records"),
            "note": "Pearson r; small n, descriptive only",
        },
        table,
    )
    return {"n_calls": len(df)}
