"""Shared fixtures."""

from __future__ import annotations

import pytest

from callgraph.config import AppConfig, get_config
from callgraph.schemas import CallTranscript, Turn


@pytest.fixture(scope="session")
def config() -> AppConfig:
    return get_config()


@pytest.fixture()
def call() -> CallTranscript:
    def t(spk: str, role: str, s: float, e: float, text: str) -> Turn:
        return Turn(speaker=spk, role=role, start=s, end=e, text=text)

    return CallTranscript(
        call_id="t1",
        duration_s=60.0,
        turns=[
            t("A", "agent", 0.0, 4.0, "Thank you for calling how can I help"),
            t("B", "customer", 3.0, 9.0, "My card is not working this is ridiculous"),
            t("A", "agent", 15.0, 20.0, "I am sorry to hear that let me verify your identity"),
            t("B", "customer", 20.5, 22.0, "Thanks"),
        ],
    )
