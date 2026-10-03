"""Extraction evaluation: field precision on the (silver) extraction check set."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from callgraph.config import AppConfig
from callgraph.silver import labels_path
from eval.common import save_result


def run(config: AppConfig) -> dict[str, Any]:
    p = labels_path(config, "extraction_check")
    if not p.exists():
        return {"note": "run scripts/silver_labels.py extraction first"}
    data = json.loads(p.read_text())
    df = pd.DataFrame(data["items"])
    per_field = (
        df.groupby("field")["final"].mean().reset_index().rename(columns={"final": "precision"})
    )
    out = {
        "overall_precision": float(df["final"].mean()),
        "n_fields": len(df),
        "n_calls": int(df.call_id.nunique()),
        "per_field": per_field.to_dict("records"),
        "cohen_kappa_annotators": data["cohen_kappa"],
        "disclaimer": data["disclaimer"],
    }
    save_result(config, "extraction_precision", out, per_field)
    return out
