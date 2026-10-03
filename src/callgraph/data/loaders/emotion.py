"""Speech-emotion datasets: CREMA-D, RAVDESS (HF parquet mirrors) and MELD (audio + text)."""

from __future__ import annotations

import io
import re
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf
from loguru import logger

from callgraph.data.download import ensure_space, hf_download
from callgraph.data.labelmaps import map_label
from callgraph.data.manifest import assign_group_splits, write_manifest
from callgraph.registry import DATASETS, DatasetLoader


class ParquetEmotionLoader(DatasetLoader):
    """Base: parquet shards with (file, audio{bytes}, emotion) -> wav files + manifest."""

    @property
    def raw(self) -> Path:
        return self.config.root / self.spec["local_path"]

    def download(self) -> None:
        src = self.spec["source"]
        ensure_space(self.config, self.raw, float(src["expected_gb"]))
        for s in src["shards"]:
            hf_download(self.config, src["repo"], s, self.raw / "parquet")

    def keep(self, filename: str) -> bool:
        return True

    def build_manifest(self) -> Path:
        wavdir = self.raw / "wav"
        wavdir.mkdir(parents=True, exist_ok=True)
        frames = [pd.read_parquet(self.raw / "parquet" / s) for s in self.spec["source"]["shards"]]
        df = pd.concat(frames).drop_duplicates("file")
        df = df[df["file"].map(lambda f: self.keep(Path(f).name))]
        items: list[dict[str, Any]] = []
        limit = int(self.spec["subset_size"])
        for _, r in (
            df.sample(n=limit, random_state=int(self.config.split.seed))
            if 0 < limit < len(df)
            else df
        ).iterrows():
            name = Path(r["file"]).stem
            wav = wavdir / f"{name}.wav"
            if not wav.exists():
                data, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
                sf.write(str(wav), data, sr)
            m = re.search(self.spec["source"]["actor_regex"], name)
            items.append(
                {
                    "id": f"{self.name}_{name}",
                    "wav": wav,
                    "actor": m.group(1) if m else name,
                    "src": r["emotion"],
                }
            )
        splits = assign_group_splits(
            sorted({i["actor"] for i in items}), dict(self.config.split.ratios), int(self.config.split.seed)  # type: ignore[attr-defined]
        )
        rows = []
        for i in items:
            rows.append(
                {
                    "id": i["id"],
                    "audio_path": str(i["wav"].relative_to(self.config.root)),
                    "transcript_path": None,
                    "duration_s": sf.info(str(i["wav"])).duration,
                    "split": splits[i["actor"]],
                    "labels": {
                        "emotion_src": i["src"],
                        "actor": i["actor"],
                        "emotion": map_label(self.config, "emotions", self.name, i["src"]),
                    },
                }
            )
        logger.info("{}: {} clips", self.name, len(rows))
        return write_manifest(self.config, self.name, rows)


@DATASETS.register("crema_d")
class CremaDLoader(ParquetEmotionLoader):
    """CREMA-D via a parquet mirror (re-split by actor to avoid speaker leakage)."""


@DATASETS.register("ravdess")
class RavdessLoader(ParquetEmotionLoader):
    """RAVDESS speech clips only (vocal channel from config) via a parquet mirror."""

    def keep(self, filename: str) -> bool:
        parts = filename.split(".")[0].split("-")
        return parts[1:2] == [self.spec["source"]["speech_vocal_channel"]]


@DATASETS.register("meld")
class MeldLoader(DatasetLoader):
    """MELD: text + emotion CSVs (GitHub) and 16 kHz utterance audio (HF ajyy/MELD_audio)."""

    @property
    def archives(self) -> dict[str, str]:
        return dict(self.spec["source"]["archives"])

    @property
    def raw(self) -> Path:
        return self.config.root / self.spec["local_path"]

    def download(self) -> None:
        from callgraph.data.download import http_download

        src = self.spec["source"]
        ensure_space(self.config, self.raw, float(src["expected_gb"]))
        for split, arc in self.archives.items():
            hf_download(self.config, src["audio_repo"], arc, self.raw / "archives")
            http_download(
                self.config,
                src["csv_url_template"].format(name=src["csv_names"][split]),
                self.raw / "csv" / f"{split}.csv",
                expected_gb=self.spec["expected_gb"],
            )

    def _extract(self, split: str, limit: int) -> dict[str, Path]:
        """Extract at most ``limit`` wav members of a split archive."""
        out = self.raw / "wav" / split
        out.mkdir(parents=True, exist_ok=True)
        found: dict[str, Path] = {}
        with tarfile.open(
            self.raw / self.archives[split].replace("archive/", "archives/archive/"), "r:gz"
        ) as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith((".wav", ".mp3", ".flac")):
                    continue
                dest = out / Path(m.name).name
                if not dest.exists():
                    if len(found) >= limit:
                        break
                    dest.write_bytes(tf.extractfile(m).read())  # type: ignore[union-attr]
                found[dest.stem] = dest
        return found

    def build_manifest(self) -> Path:
        limit = int(self.spec["subset_size"])
        rows: list[dict[str, Any]] = []
        for split in self.archives:
            df = pd.read_csv(self.raw / "csv" / f"{split}.csv")
            wavs = self._extract(split, limit)
            for _, r in df.iterrows():
                stem = f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}"
                if stem not in wavs:
                    continue
                labels = {
                    "emotion_src": r["Emotion"],
                    "text": r["Utterance"],
                    "speaker": r["Speaker"],
                    "sentiment": r["Sentiment"],
                    "emotion": map_label(self.config, "emotions", self.name, r["Emotion"]),
                }
                rows.append(
                    {
                        "id": f"meld_{stem}",
                        "audio_path": str(wavs[stem].relative_to(self.config.root)),
                        "transcript_path": None,
                        "duration_s": sf.info(str(wavs[stem])).duration,
                        "split": split,
                        "labels": labels,
                    }
                )
        logger.info("MELD: {} utterances", len(rows))
        return write_manifest(self.config, self.name, rows)
