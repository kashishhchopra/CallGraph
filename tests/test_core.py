"""Unit tests for config, label maps, preprocessing, dynamics, schema builders and LLM helpers."""

from __future__ import annotations

import numpy as np
import pytest

from callgraph.core.preprocess import (
    normalize_for_wer,
    normalize_loudness,
    simulate_telephone,
    trim_long_silences,
)
from callgraph.data.labelmaps import UnmappedLabelError, map_label, shared_labels
from callgraph.data.manifest import assign_group_splits
from callgraph.dynamics.features import compute
from callgraph.emotion.model import frustration_score, summarize_trajectory
from callgraph.graph.library import list_queries, load_query
from callgraph.graph.schema import GraphSchema
from callgraph.llm.client import extract_json, render_prompt
from callgraph.schemas import TurnEmotion


def test_config_loads_and_env_override(config, monkeypatch):
    assert config.falkordb.graph_name
    assert config.path("data_dir").is_absolute()


def test_label_maps_are_total_and_fail_loudly(config):
    assert map_label(config, "intents", "banking77", "card_arrival") in shared_labels(
        config, "intents"
    )
    assert (
        map_label(config, "intents", "abcd", "recover_username", "account_access")
        == "account_access"
    )
    with pytest.raises(UnmappedLabelError):
        map_label(config, "intents", "banking77", "definitely_not_a_label")
    with pytest.raises(UnmappedLabelError):
        map_label(config, "emotions", "meld", "bogus")


def test_group_split_has_no_leakage():
    groups = [f"spk{i}" for i in range(100)]
    a = assign_group_splits(groups, {"train": 0.8, "validation": 0.1, "test": 0.1}, 1)
    b = assign_group_splits(groups, {"train": 0.8, "validation": 0.1, "test": 0.1}, 1)
    assert a == b and set(a.values()) == {"train", "validation", "test"}


def test_preprocessing():
    sr = 16000
    y = np.concatenate(
        [
            np.random.RandomState(0).randn(sr) * 0.1,
            np.zeros(sr * 5),
            np.random.RandomState(1).randn(sr) * 0.1,
        ]
    )
    out, removed = trim_long_silences(y, sr, 2.0, -45.0, 0.5, 0.02)
    assert len(out) < len(y) and removed
    assert abs(20 * np.log10(np.sqrt(np.mean(normalize_loudness(y[:sr], -20.0) ** 2))) + 20) < 1.0
    assert len(simulate_telephone(y[:sr], sr, 8000)) == sr
    assert normalize_for_wer("Hello, World!!") == "hello world"


def test_dynamics(config, call):
    d = compute(config, call, wait_time_s=30)
    assert d.features["interruptions"] == 1 and d.features["wait_time_s"] == 30
    assert d.features["long_silences"] >= 1
    assert 0 < d.features["agent_talk_ratio"] < 1


def test_emotion_trajectory(config):
    probs = lambda a, h: {"angry_frustrated": a, "happy": h, "neutral": 1 - a - h}  # noqa: E731
    turns = [
        TurnEmotion(turn_index=0, emotion="angry_frustrated", intensity=0.9, probs=probs(0.9, 0.0)),
        TurnEmotion(turn_index=1, emotion="happy", intensity=0.8, probs=probs(0.0, 0.8)),
    ]
    out = summarize_trajectory(config, "c", turns, ["this is ridiculous, get me a manager"])
    assert out.shift == "improved" and not out.ends_negative and 0 <= out.frustration_score <= 100
    assert frustration_score(config, turns, [""])[1] == pytest.approx(0.9)


def test_graph_schema_builds_parameterised_merges(config):
    s = GraphSchema(config)
    q, p = s.merge_node("Call", {"call_id": "x", "dataset": "d", "bogus": 1})
    assert "$key" in q and "bogus" not in q and p["key"] == "x"
    q, p = s.merge_rel("HAS_INTENT", "x", "refund", {"confidence": 0.5})
    assert "HAS_INTENT" in q and p["dst"] == "refund"
    assert "(:Call)-[:HAS_INTENT" in s.schema_text()
    with pytest.raises(KeyError):
        s.merge_node("Call", {"dataset": "d"})


def test_query_library_is_parameterised():
    for name in list_queries():
        cypher, params, _ = load_query(name)
        assert all(f"${p}" in cypher for p in params), name


def test_llm_helpers(config):
    assert extract_json('Sure! ```json\n{"a": 1}\n``` done') == {"a": 1}
    assert extract_json('text {"a": [1, 2]} tail') == {"a": [1, 2]}
    prompt = render_prompt("judge", config, question="q", reference="r", answer="a")
    assert "Reference answer: r" in prompt
