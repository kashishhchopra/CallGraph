"""SpokenWOZ test split (audio + timed text) from the Hugging Face mirrors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import soundfile as sf
from loguru import logger

from callgraph.data.download import ensure_space, hf_download
from callgraph.data.labelmaps import map_label
from callgraph.data.manifest import write_manifest
from callgraph.registry import DATASETS, DatasetLoader

ROLE = {"user": "customer", "system": "agent"}
MS_PER_S = 1000.0


@DATASETS.register("spokenwoz")
class SpokenWozLoader(DatasetLoader):
    """SpokenWOZ: task-oriented spoken dialogues with word timings and domains."""

    @property
    def raw(self) -> Path:
        return self.config.root / self.spec["local_path"]

    def _ids(self) -> list[str]:
        lines = (self.raw / "text" / self.spec["source"]["list_file"]).read_text().split()
        ids = [x for x in lines if x]
        return ids[: int(self.spec["subset_size"])]

    def download(self) -> None:
        src = self.spec["source"]
        hf_download(self.config, src["text_repo"], src["text_file"], self.raw / "text")
        hf_download(self.config, src["text_repo"], src["list_file"], self.raw / "text")
        ids = self._ids()
        ensure_space(self.config, self.raw, float(src["audio_gb_per_dialogue"]) * len(ids))
        for did in ids:
            hf_download(
                self.config, src["audio_repo"], f"{did}{src['audio_ext']}", self.raw / "audio"
            )

    def build_manifest(self) -> Path:
        data = json.loads((self.raw / "text" / self.spec["source"]["text_file"]).read_text())
        rows: list[dict[str, Any]] = []
        for did in self._ids():
            wav = self.raw / "audio" / f"{did}{self.spec['source']['audio_ext']}"
            if did not in data or not wav.exists():
                continue
            dlg = data[did]
            turns = []
            for t in dlg["log"]:
                words = t.get("words") or []
                if not words:
                    continue
                turns.append(
                    {
                        "speaker": t["tag"],
                        "role": ROLE[t["tag"]],
                        "text": t["text"],
                        "start": min(w["BeginTime"] for w in words) / MS_PER_S,
                        "end": max(w["EndTime"] for w in words) / MS_PER_S,
                        "words": [
                            {
                                "text": w["Word"],
                                "start": w["BeginTime"] / MS_PER_S,
                                "end": w["EndTime"] / MS_PER_S,
                            }
                            for w in words
                        ],
                    }
                )
            domains = [k for k, v in dlg["goal"].items() if v and k not in ("message", "topic")]
            main = next((d for d in domains if d != "profile"), domains[0])
            labels = {
                "domains": domains,
                "intent": map_label(self.config, "intents", self.name, main),
            }
            tdir = self.config.path("processed_dir") / self.name / "transcripts"
            tdir.mkdir(parents=True, exist_ok=True)
            tpath = tdir / f"{did}.json"
            tpath.write_text(json.dumps({"id": did, "turns": turns, "labels": labels}))
            rows.append(
                {
                    "id": did,
                    "audio_path": str(wav.relative_to(self.config.root)),
                    "transcript_path": str(tpath.relative_to(self.config.root)),
                    "duration_s": sf.info(str(wav)).duration,
                    "split": "test",
                    "labels": labels,
                }
            )
        logger.info("SpokenWOZ: {} dialogues", len(rows))
        return write_manifest(self.config, self.name, rows)
