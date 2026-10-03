"""Text datasets: ABCD, Banking77, DialogSum, TweetSumm."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from callgraph.data.download import hf_download, http_download
from callgraph.data.labelmaps import map_label
from callgraph.data.manifest import write_manifest
from callgraph.registry import DATASETS, DatasetLoader


class TextLoader(DatasetLoader):
    """Shared helpers for text-only datasets."""

    @property
    def raw(self) -> Path:
        return self.config.root / self.spec["local_path"]

    @property
    def processed(self) -> Path:
        return self.config.path("processed_dir") / self.name

    def write_dialogue(self, did: str, turns: list[dict[str, str]], labels: dict[str, Any]) -> Path:
        """Persist one dialogue as JSON under data/processed/<name>/transcripts/."""
        path = self.processed / "transcripts" / f"{did}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"id": did, "turns": turns, "labels": labels}, ensure_ascii=False)
        )
        return path


@DATASETS.register("abcd")
class AbcdLoader(TextLoader):
    """ABCD: 10k text dialogues with flows/subflows, actions and guidelines."""

    def download(self) -> None:
        src = self.spec["source"]
        for f in src["files"]:
            http_download(
                self.config,
                f"{src['base_url']}/{f}",
                self.raw / f,
                expected_gb=self.spec["expected_gb"],
            )

    @staticmethod
    def norm_action(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    def required_actions(self) -> dict[tuple[str, str], list[str]]:
        """(flow, subflow) snake_case -> required (non-communication) action names."""
        guide = json.loads((self.raw / self.spec["source"]["guidelines_file"]).read_text())
        out: dict[tuple[str, str], list[str]] = {}
        snake = lambda s: re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")  # noqa: E731
        for flow, fv in guide.items():
            for sub, sv in fv["subflows"].items():
                out[(snake(flow), snake(sub))] = [
                    self.norm_action(a["button"]) for a in sv["actions"] if a["button"] != "N/A"
                ]
        return out

    def iter_conversations(self) -> dict[str, list[dict[str, Any]]]:
        return json.load(gzip.open(self.raw / self.spec["source"]["main_file"]))

    def build_manifest(self) -> Path:
        data = self.iter_conversations()
        required = self.required_actions()
        sub = int(self.spec["subset_size"])
        out: list[dict[str, Any]] = []
        for split, convs in data.items():
            for conv in convs[: len(convs) if split == "train" else sub]:
                sc = conv["scenario"]
                flow, subflow = sc["flow"], sc["subflow"]
                performed = [
                    t["targets"][2]
                    for t in conv["delexed"]
                    if t["speaker"] == "action" and t["targets"][1] == "take_action"
                ]
                labels = {
                    "flow": flow,
                    "subflow": subflow,
                    "intent": map_label(self.config, "intents", self.name, subflow, flow),
                    "required_actions": required.get((flow, subflow), []),
                    "performed_actions": performed,
                }
                did = f"abcd_{conv['convo_id']}"
                turns = [{"speaker": s, "role": s, "text": t} for s, t in conv["original"]]
                path = self.write_dialogue(did, turns, labels)
                out.append(
                    {
                        "id": did,
                        "audio_path": None,
                        "transcript_path": str(path.relative_to(self.config.root)),
                        "duration_s": None,
                        "split": split,
                        "labels": labels,
                    }
                )
        logger.info("ABCD: {} dialogues", len(out))
        return write_manifest(self.config, self.name, out)


@DATASETS.register("banking77")
class Banking77Loader(TextLoader):
    """Banking77 single-turn intent queries."""

    def download(self) -> None:
        src = self.spec["source"]
        for f in src["files"]:
            http_download(
                self.config,
                f"{src['base_url']}/{f}",
                self.raw / f,
                expected_gb=self.spec["expected_gb"],
            )

    def build_manifest(self) -> Path:
        rows = []
        for split in self.spec["splits"]:
            df = pd.read_csv(self.raw / f"{split}.csv")
            for i, r in df.iterrows():
                did = f"banking77_{split}_{i}"
                labels = {
                    "source": r["category"],
                    "intent": map_label(self.config, "intents", self.name, r["category"]),
                }
                path = self.write_dialogue(
                    did, [{"speaker": "customer", "role": "customer", "text": r["text"]}], labels
                )
                rows.append(
                    {
                        "id": did,
                        "audio_path": None,
                        "transcript_path": str(path.relative_to(self.config.root)),
                        "duration_s": None,
                        "split": split,
                        "labels": labels,
                    }
                )
        return write_manifest(self.config, self.name, rows)


@DATASETS.register("dialogsum")
class DialogSumLoader(TextLoader):
    """DialogSum human-summarised dialogues."""

    def download(self) -> None:
        for s in self.spec["splits"]:
            http_download(
                self.config,
                self.spec["source"]["url_template"].format(split=s),
                self.raw / f"{s}.csv",
                expected_gb=self.spec["expected_gb"],
            )

    def build_manifest(self) -> Path:
        rows = []
        for split in self.spec["splits"]:
            df = pd.read_csv(self.raw / f"{split}.csv")
            for _, r in df.iterrows():
                turns = []
                for line in str(r["dialogue"]).split("\n"):
                    m = re.match(r"#(Person\d)#:\s*(.*)", line)
                    if m:
                        turns.append(
                            {"speaker": m.group(1), "role": m.group(1), "text": m.group(2)}
                        )
                labels = {"summary": r["summary"], "topic": r.get("topic")}
                path = self.write_dialogue(str(r["id"]), turns, labels)
                rows.append(
                    {
                        "id": str(r["id"]),
                        "audio_path": None,
                        "transcript_path": str(path.relative_to(self.config.root)),
                        "duration_s": None,
                        "split": split,
                        "labels": labels,
                    }
                )
        return write_manifest(self.config, self.name, rows)


@DATASETS.register("tweetsumm")
class TweetSummLoader(TextLoader):
    """TweetSumm via the community text mirror (official release needs Kaggle's twcs.csv)."""

    def download(self) -> None:
        src = self.spec["source"]
        for f in src["files"].values():
            hf_download(self.config, src["repo"], f, self.raw)

    def build_manifest(self) -> Path:
        rows = []
        for split, f in self.spec["source"]["files"].items():
            df = pd.read_parquet(self.raw / f)
            for i, r in df.iterrows():
                turns = []
                for line in str(r["conversation"]).strip().split("\n"):
                    who, _, text = line.partition(": ")
                    role = "customer" if who == "user" else "agent"
                    turns.append({"speaker": role, "role": role, "text": text})
                did = f"tweetsumm_{split}_{i}"
                labels = {"summary": r["summary"]}
                path = self.write_dialogue(did, turns, labels)
                rows.append(
                    {
                        "id": did,
                        "audio_path": None,
                        "transcript_path": str(path.relative_to(self.config.root)),
                        "duration_s": None,
                        "split": split,
                        "labels": labels,
                    }
                )
        return write_manifest(self.config, self.name, rows)
