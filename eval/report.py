"""Build the Markdown results report from saved run files only (no numbers are typed here)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from callgraph.config import get_config
from callgraph.logging_setup import setup_logging
from eval.common import run_dir


def table_md(rows: list[dict], fmt: str) -> str:
    if not rows:
        return "_no rows_"
    return pd.DataFrame(rows).to_markdown(
        index=False, floatfmt=fmt.replace("%", "").replace("f", "f")
    )


def main() -> Path:
    setup_logging()
    cfg = get_config()
    rep = cfg.evaluation.report  # type: ignore[attr-defined]
    d = run_dir(cfg)
    out = [
        f"# {rep['title']}",
        "",
        f"Run: `{d.name}`. Every number below was read from files in `outputs/eval/{d.name}/`.",
        "",
    ]
    for sec in rep["sections"]:
        p = d / f"{sec['file']}.json"
        out += [f"## {sec['title']}", ""]
        if not p.exists():
            out += ["_not run in this evaluation (see run_status / error files)_", ""]
            continue
        data = json.loads(p.read_text())
        if isinstance(data.get("rows"), list):
            out += [table_md(data["rows"], rep["float_format"]), ""]
        scalars = {k: v for k, v in data.items() if k != "rows" and not isinstance(v, (list, dict))}
        out += [f"- {k}: {v}" for k, v in scalars.items()]
        for k, v in data.items():
            if isinstance(v, list) and k != "rows" and v and isinstance(v[0], dict):
                out += ["", f"**{k}**", "", table_md(v, rep["float_format"])]
            elif isinstance(v, dict):
                out += [f"- {k}: `{json.dumps(v, default=str)}`"]
        png = d / f"{sec['file']}.png"
        if png.exists():
            out += ["", f"![{sec['title']}]({png.name})"]
        out.append("")
    for png in sorted(d.glob("*.png")):
        if not any(png.stem == s["file"] for s in rep["sections"]):
            out += [f"![{png.stem}]({png.name})", ""]
    notes = rep.get("notes", [])
    if notes:
        out += ["## Notes and caveats", ""] + [f"- {n}" for n in notes] + [""]
    path = d / "report.md"
    path.write_text("\n".join(out))
    return path


if __name__ == "__main__":
    print(main())
