"""UniData call-center sample (real calls). Local use only; never redistribute."""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import soundfile as sf
from huggingface_hub import HfApi
from loguru import logger

from callgraph.data.download import hf_download
from callgraph.data.manifest import write_manifest
from callgraph.registry import DATASETS, DatasetLoader

LINE = re.compile(
    r"\[(?P<spk>Speaker [^\]]+)\]\s*\[(?P<h>\d+):(?P<m>\d+):(?P<s>\d+)\]:\s*(?P<text>.*)"
)


def parse_transcript(text: str) -> list[dict[str, Any]]:
    """Parse '[Speaker 1] [00:00:08]: text' lines into timed turns (start only)."""
    turns = []
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m:
            start = timedelta(
                hours=int(m["h"]), minutes=int(m["m"]), seconds=int(m["s"])
            ).total_seconds()
            turns.append({"speaker": m["spk"], "start": start, "text": m["text"]})
    return turns


@DATASETS.register("unidata")
class UniDataLoader(DatasetLoader):
    """HF UniDataPro/call-center-audio: few flac files + metadata for ~890 transcripts."""

    @property
    def raw(self) -> Path:
        return self.config.root / self.spec["local_path"]

    def download(self) -> None:
        files = [
            s.rfilename
            for s in HfApi().dataset_info(self.spec["source"]["repo"]).siblings
            if s.rfilename.startswith(self.spec["source"]["files_prefix"])
        ]
        for f in files:
            hf_download(self.config, self.spec["source"]["repo"], f, self.raw)

    def build_manifest(self) -> Path:
        meta = (self.raw / self.spec["source"]["metadata_file"]).read_text(encoding="utf-8-sig")
        rows = []
        for r in csv.DictReader(io.StringIO(meta), delimiter=self.spec["source"]["csv_delimiter"]):
            cid = r["call_id"]
            flac = self.raw / self.spec["source"]["files_prefix"] / f"{cid}.flac"
            audio = str(flac.relative_to(self.config.root)) if flac.exists() else None
            dur = (
                sf.info(str(flac)).duration
                if audio
                else float(r["Total_call_duration_in_seconds"] or 0)
            )
            tdir = self.config.path("processed_dir") / self.name / "transcripts"
            tdir.mkdir(parents=True, exist_ok=True)
            tpath = tdir / f"{cid}.json"
            tpath.write_text(json.dumps({"id": cid, "turns": parse_transcript(r["Transcription"])}))
            wait = r["representative_waiting_time_in_seconds"]
            rows.append(
                {
                    "id": cid,
                    "audio_path": audio,
                    "transcript_path": str(tpath.relative_to(self.config.root)),
                    "duration_s": dur,
                    "split": "test",
                    "labels": {
                        "category": r["Category"],
                        "company": r["Company_name"],
                        "date": r["date_of_call"],
                        "wait_time_s": float(wait) if wait else None,
                        "ai_summary": r["AI Sumary"],
                        "has_audio": audio is not None,
                    },
                }
            )
        logger.info(
            "UniData: {} rows, {} with audio", len(rows), sum(1 for r in rows if r["audio_path"])
        )
        return write_manifest(self.config, self.name, rows)
