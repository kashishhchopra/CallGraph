"""Provider-agnostic LLM client (Ollama, OpenAI-compatible, Anthropic).

Features: retry with backoff, timeout, JSON mode, on-disk response cache.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from loguru import logger
from pydantic import BaseModel, ValidationError

from callgraph.config import AppConfig, get_config

PROVIDERS = ("ollama", "openai", "anthropic")


class LLMError(RuntimeError):
    """Raised when the LLM call fails after retries."""


def render_prompt(name: str, config: AppConfig | None = None, **variables: Any) -> str:
    """Render ``prompts/<name>.j2`` with the given variables."""
    cfg = config or get_config()
    env = Environment(
        loader=FileSystemLoader(str(cfg.path("prompts_dir"))),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(f"{name}.j2").render(**variables)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating code fences and prose around it."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
        if start < 0:
            raise
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        end = text.rfind(closer)
        return json.loads(text[start : end + 1])


class LLMClient:
    """Chat/completion client driven entirely by ``config.llm``."""

    def __init__(self, config: AppConfig | None = None, model: str | None = None) -> None:
        self.config = config or get_config()
        self.cfg = self.config.llm
        self.model = model or self.cfg.model
        if self.cfg.provider not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider {self.cfg.provider}; use one of {PROVIDERS}")
        self.cache_dir: Path = self.config.path("cache_dir") / str(self.cfg.cache_subdir)
        if self.cfg.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ cache
    def _key(self, prompt: str, system: str | None, json_mode: bool) -> Path:
        blob = json.dumps(
            [self.cfg.provider, self.model, prompt, system, json_mode, self.cfg.temperature]
        )
        return self.cache_dir / (hashlib.sha256(blob.encode()).hexdigest() + ".json")

    # ------------------------------------------------------------- providers
    def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        resp = httpx.post(url, json=payload, headers=headers, timeout=self.cfg.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def _call(self, prompt: str, system: str | None, json_mode: bool) -> str:
        base = self.cfg.base_url.rstrip("/")
        key = self.config.llm_api_key.get_secret_value() if self.config.llm_api_key else ""
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        if self.cfg.provider == "ollama":
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": msgs,
                "stream": False,
                "keep_alive": self.cfg.keep_alive,
                "options": {
                    "temperature": self.cfg.temperature,
                    "num_ctx": self.cfg.num_ctx,
                    "num_predict": self.cfg.max_tokens,
                },
            }
            if json_mode:
                payload["format"] = "json"
            return self._post(f"{base}/api/chat", payload, {})["message"]["content"]
        if self.cfg.provider == "openai":
            payload = {"model": self.model, "messages": msgs, "temperature": self.cfg.temperature}
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            out = self._post(
                f"{base}/v1/chat/completions", payload, {"Authorization": f"Bearer {key}"}
            )
            return out["choices"][0]["message"]["content"]
        payload = {
            "model": self.model,
            "max_tokens": self.cfg.anthropic_max_tokens,
            "temperature": self.cfg.temperature,
            "messages": [m for m in msgs if m["role"] != "system"],
        }
        if system:
            payload["system"] = system
        out = self._post(
            f"{base}/v1/messages",
            payload,
            {"x-api-key": key, "anthropic-version": str(self.cfg.anthropic_version)},
        )
        return "".join(b.get("text", "") for b in out["content"])

    # ------------------------------------------------------------- public API
    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        """Return the model's text reply (cached, with retry/backoff)."""
        use_json = json_mode and bool(self.cfg.json_mode)
        path = self._key(prompt, system, use_json)
        if self.cfg.cache_enabled and path.exists():
            return json.loads(path.read_text())["text"]
        last: Exception | None = None
        for attempt in range(1, int(self.cfg.max_retries) + 1):
            try:
                text = self._call(prompt, system, use_json)
                if self.cfg.cache_enabled:
                    path.write_text(json.dumps({"text": text}))
                return text
            except (httpx.HTTPError, KeyError) as exc:
                last = exc
                logger.warning("LLM call failed (attempt {}): {}", attempt, exc)
                time.sleep(float(self.cfg.retry_backoff_s) * attempt)
        raise LLMError(f"LLM call failed after {self.cfg.max_retries} attempts: {last}")

    def complete_json(self, prompt: str, system: str | None = None) -> Any:
        """Return parsed JSON from the model."""
        return extract_json(self.complete(prompt, system, json_mode=True))

    def complete_model(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str | None = None,
        max_retries: int | None = None,
    ) -> BaseModel:
        """Return a validated Pydantic object; on invalid output retry with the error appended."""
        tries = int(max_retries if max_retries is not None else self.cfg.max_retries)
        feedback = ""
        last: Exception | None = None
        for _ in range(tries):
            raw = self.complete(prompt + feedback, system, json_mode=True)
            try:
                return schema.model_validate(extract_json(raw))
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                last = exc
                feedback = (
                    f"\n\nYour previous reply was invalid ({str(exc)[: int(self.cfg.error_trunc_chars)]}). "
                    "Reply again with ONLY valid JSON matching the schema."
                )
        raise LLMError(f"No valid {schema.__name__} after {tries} tries: {last}")

    def warm(self) -> None:
        """Ask the provider to load the model into memory (Ollama: an empty prompt with keep-alive)."""
        if self.cfg.provider == "ollama":
            httpx.post(
                f"{self.cfg.base_url.rstrip('/')}/api/generate",
                json={"model": self.model, "keep_alive": self.cfg.keep_alive},
                timeout=self.cfg.timeout_s,
            ).raise_for_status()

    def health(self) -> bool:
        """True if the provider endpoint responds."""
        try:
            if self.cfg.provider == "ollama":
                httpx.get(
                    f"{self.cfg.base_url.rstrip('/')}/api/tags",
                    timeout=float(self.cfg.health_timeout_s),
                ).raise_for_status()
            return True
        except httpx.HTTPError:
            return False
