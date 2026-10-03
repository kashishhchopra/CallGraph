"""Diarization/meeting corpora: AMI, VoxConverse, Earnings-22 (ASR), CallHome (manual), AxonData (probe)."""

from __future__ import annotations

import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import httpx
import soundfile as sf
from loguru import logger

from callgraph.data.download import ensure_space, hf_download, http_download
from callgraph.data.manifest import write_manifest
from callgraph.registry import DATASETS, DatasetLoader


class BaseLoader(DatasetLoader):
    """Shared path helpers."""

    @property
    def raw(self) -> Path:
        return self.config.root / self.spec["local_path"]

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.config.root))

    def tdir(self) -> Path:
        d = self.config.path("processed_dir") / self.name / "transcripts"
        d.mkdir(parents=True, exist_ok=True)
        return d


def parse_rttm(text: str, fmt: dict[str, Any]) -> list[dict[str, Any]]:
    """RTTM lines -> [{speaker, start, end}] (field positions come from ``settings.yaml``)."""
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) >= int(fmt["min_fields"]) and f[0] == fmt["type_label"]:
            start = float(f[int(fmt["start_idx"])])
            out.append(
                {
                    "speaker": f[int(fmt["speaker_idx"])],
                    "start": start,
                    "end": start + float(f[int(fmt["duration_idx"])]),
                }
            )
    return out


@DATASETS.register("ami")
class AmiLoader(BaseLoader):
    """AMI meetings: headset-mix audio, RTTM (diarization + overlaps), word-level turns, abstractive summaries."""

    def meetings(self) -> list[str]:
        return list(self.spec["source"]["meetings"])[: int(self.spec["subset_size"])]

    def download(self) -> None:
        src = self.spec["source"]
        ensure_space(self.config, self.raw, float(self.spec["expected_gb"]))
        http_download(
            self.config,
            src["annotations_url"],
            self.raw / "annotations.zip",
            expected_gb=self.spec["expected_gb"],
        )
        for m in self.meetings():
            http_download(
                self.config,
                src["audio_url"].format(m=m),
                self.raw / "audio" / f"{m}.wav",
                expected_gb=self.spec["expected_gb"],
            )
            http_download(
                self.config,
                src["rttm_url"].format(m=m),
                self.raw / "rttm" / f"{m}.rttm",
                expected_gb=float(self.config.downloads.small_file_gb),
            )

    def _turns(self, zf: zipfile.ZipFile, m: str) -> list[dict[str, Any]]:
        src = self.spec["source"]
        pat = re.compile(
            src["words_glob"].format(m=re.escape(m)).replace("*", r"([A-Z])").replace(".", r"\.")
        )
        words = []
        for name in zf.namelist():
            mt = pat.fullmatch(name)
            if not mt:
                continue
            root = ET.fromstring(zf.read(name))
            for w in root.iter():
                if w.tag.endswith("w") and w.get("starttime") and w.text:
                    words.append(
                        {
                            "speaker": mt.group(1),
                            "start": float(w.get("starttime")),
                            "end": float(w.get("endtime")),
                            "text": w.text,
                        }
                    )
        words.sort(key=lambda w: w["start"])
        turns: list[dict[str, Any]] = []
        for w in words:
            last = turns[-1] if turns else None
            if (
                last
                and last["speaker"] == w["speaker"]
                and w["start"] - last["end"] < float(src["turn_gap_s"])
            ):
                last["end"] = w["end"]
                last["text"] += " " + w["text"]
            else:
                turns.append(
                    {
                        "speaker": w["speaker"],
                        "role": w["speaker"],
                        "start": w["start"],
                        "end": w["end"],
                        "text": w["text"],
                    }
                )
        return turns

    def _summary(self, zf: zipfile.ZipFile, m: str) -> str:
        path = self.spec["source"]["summary_path"].format(m=m)
        if path not in zf.namelist():
            return ""
        root = ET.fromstring(zf.read(path))
        abstract = next((e for e in root.iter() if e.tag.endswith("abstract")), None)
        return (
            " ".join((s.text or "").strip() for s in abstract.iter() if s.tag.endswith("sentence"))
            if abstract is not None
            else ""
        )

    def build_manifest(self) -> Path:
        rows = []
        with zipfile.ZipFile(self.raw / "annotations.zip") as zf:
            for m in self.meetings():
                wav = self.raw / "audio" / f"{m}.wav"
                rttm = (self.raw / "rttm" / f"{m}.rttm").read_text()
                turns = self._turns(zf, m)
                summary = self._summary(zf, m)
                tp = self.tdir() / f"{m}.json"
                tp.write_text(
                    json.dumps(
                        {
                            "id": m,
                            "turns": turns,
                            "rttm": parse_rttm(rttm, dict(self.config.rttm.model_extra or {})),
                            "labels": {"summary": summary},
                        }
                    )
                )
                rows.append(
                    {
                        "id": m,
                        "audio_path": self.rel(wav),
                        "transcript_path": self.rel(tp),
                        "duration_s": sf.info(str(wav)).duration,
                        "split": "test",
                        "labels": {
                            "rttm": self.rel(self.raw / "rttm" / f"{m}.rttm"),
                            "summary": summary,
                        },
                    }
                )
        logger.info("AMI: {} meetings", len(rows))
        return write_manifest(self.config, self.name, rows)


@DATASETS.register("voxconverse")
class VoxConverseLoader(BaseLoader):
    """VoxConverse test: RTTM from GitHub, selected wavs extracted from the remote zip by range requests."""

    def _names(self) -> list[str]:
        src = self.spec["source"]
        listing = httpx.get(src["rttm_api"].format(split=src["split"]), timeout=60).json()
        return sorted(x["name"] for x in listing if x["name"].endswith(".rttm"))

    def download(self) -> None:
        from remotezip import RemoteZip

        src = self.spec["source"]
        want = int(self.spec["subset_size"])
        limit = int(src["max_file_mb"]) * int(self.config.downloads.bytes_per_mb)
        ensure_space(self.config, self.raw, float(self.spec["expected_gb"]))
        prefix = src["zip_prefix"].format(split=src["split"])
        got = 0
        with RemoteZip(src["zip_url"].format(split=src["split"])) as z:
            info = {i.filename: i for i in z.infolist()}
            for name in self._names():
                stem = Path(name).stem
                member = f"{prefix}{stem}.wav"
                if member not in info or info[member].file_size > limit:
                    continue
                dest = self.raw / "audio" / f"{stem}.wav"
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(z.read(member))
                http_download(
                    self.config,
                    src["rttm_raw"].format(split=src["split"], name=name),
                    self.raw / "rttm" / name,
                    expected_gb=float(self.config.downloads.small_file_gb),
                )
                got += 1
                if got >= want:
                    break

    def build_manifest(self) -> Path:
        rows = []
        for wav in sorted((self.raw / "audio").glob("*.wav")):
            rttm = self.raw / "rttm" / f"{wav.stem}.rttm"
            rows.append(
                {
                    "id": wav.stem,
                    "audio_path": self.rel(wav),
                    "transcript_path": None,
                    "duration_s": sf.info(str(wav)).duration,
                    "split": "test",
                    "labels": {"rttm": self.rel(rttm)},
                }
            )
        return write_manifest(self.config, self.name, rows)


@DATASETS.register("earnings22")
class Earnings22Loader(BaseLoader):
    """Earnings-22 ASR: first rows of one parquet shard -> wav + reference text."""

    def download(self) -> None:
        src = self.spec["source"]
        ensure_space(self.config, self.raw, float(self.spec["expected_gb"]))
        hf_download(self.config, src["repo"], src["shard"], self.raw / "parquet")

    def build_manifest(self) -> Path:
        import pandas as pd

        src = self.spec["source"]
        df = pd.read_parquet(self.raw / "parquet" / src["shard"]).head(
            int(self.spec["subset_size"])
        )
        rows = []
        for i, r in df.reset_index(drop=True).iterrows():
            did = f"earnings22_{i}"
            wav = self.raw / "wav" / f"{did}.wav"
            if not wav.exists():
                wav.parent.mkdir(parents=True, exist_ok=True)
                data, sr = sf.read(io.BytesIO(r[src["audio_col"]]["bytes"]))
                sf.write(str(wav), data, sr)
            tp = self.tdir() / f"{did}.json"
            tp.write_text(json.dumps({"id": did, "reference": r[src["text_col"]], "turns": []}))
            rows.append(
                {
                    "id": did,
                    "audio_path": self.rel(wav),
                    "transcript_path": self.rel(tp),
                    "duration_s": sf.info(str(wav)).duration,
                    "split": "test",
                    "labels": {},
                }
            )
        return write_manifest(self.config, self.name, rows)


@DATASETS.register("callhome")
class CallHomeLoader(BaseLoader):
    """CallHome English needs a TalkBank account: print instructions; build manifest if files are present."""

    def download(self) -> None:
        logger.warning("CallHome requires manual download. {}", self.spec["source"]["instructions"])

    def build_manifest(self) -> Path:
        src = self.spec["source"]
        audio = (
            sorted((self.raw / src["audio_dir"]).glob("*"))
            if (self.raw / src["audio_dir"]).exists()
            else []
        )
        if not audio:
            logger.warning("CallHome files not found in {}; skipping (see README).", self.raw)
        rows = [
            {
                "id": a.stem,
                "audio_path": self.rel(a),
                "transcript_path": None,
                "duration_s": sf.info(str(a)).duration,
                "split": "test",
                "labels": {"cha": self.rel(self.raw / src["cha_dir"] / f"{a.stem}.cha")},
            }
            for a in audio[: int(self.spec["subset_size"])]
        ]
        return write_manifest(self.config, self.name, rows)


@DATASETS.register("axondata")
class AxonDataLoader(BaseLoader):
    """AxonData call-center sample: probes the Hub repo; builds a manifest only if audio files exist."""

    def download(self) -> None:
        from huggingface_hub import HfApi

        files = [s.rfilename for s in HfApi().dataset_info(self.spec["source"]["repo"]).siblings]
        audio = [f for f in files if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))]
        logger.info("AxonData repo files: {} (audio: {})", files, audio)
        for f in audio[: int(self.spec["subset_size"]) or len(audio)]:
            hf_download(self.config, self.spec["source"]["repo"], f, self.raw)

    def build_manifest(self) -> Path:
        audio = sorted(
            p for p in self.raw.rglob("*") if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")
        )
        if not audio:
            logger.warning("AxonData: no audio files published in the public repo (README only)")
        rows = []
        for a in audio:
            m = re.match(
                r"(?P<lang>\S+)\s*(?P<domain>.*)", a.parent.name.split("(", 1)[-1].rstrip(") ")
            )
            rows.append(
                {
                    "id": f"axon_{a.stem}",
                    "audio_path": self.rel(a),
                    "transcript_path": None,
                    "duration_s": sf.info(str(a)).duration,
                    "split": "test",
                    "labels": {
                        "language_hint": m["lang"] if m else None,
                        "domain": m["domain"].strip() if m else None,
                    },
                }
            )
        return write_manifest(self.config, self.name, rows)
