"""Fails on magic numbers and hard-coded URLs/model ids/paths in library code (config belongs in configs/)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN = ["src", "api"]
ALLOWED_NUMBERS = {0, 1, 2, -1, 10, 20, 100, 1000, 60}  # identities / unit conversions only
TINY = 1e-6  # numerical epsilons
URL = re.compile(r"https?://")
MODEL_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MODEL_WORD = re.compile(r"(whisper|bert|roberta|wav2vec|hubert|llama|bart|flan|minilm)", re.I)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
            ):
                out.add(id(body[0].value))
    return out


def violations() -> list[str]:
    found: list[str] = []
    for d in SCAN:
        for path in (ROOT / d).rglob("*.py"):
            tree = ast.parse(path.read_text())
            skip = _docstring_nodes(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or id(node) in skip:
                    continue
                v = node.value
                where = f"{path.relative_to(ROOT)}:{node.lineno}"
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)) and v not in ALLOWED_NUMBERS and abs(v) > TINY:
                    found.append(f"{where} magic number {v!r}")
                elif isinstance(v, str) and (
                    URL.search(v)
                    or (MODEL_ID.match(v) and "." not in v.split("/")[0] and MODEL_WORD.search(v))
                ):
                    found.append(f"{where} hard-coded URL/model id {v!r}")
    return found


def test_no_magic_numbers_or_urls() -> None:
    bad = violations()
    assert not bad, "Hard-coded values found:\n" + "\n".join(bad)
