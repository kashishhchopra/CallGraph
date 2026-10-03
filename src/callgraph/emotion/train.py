"""M3 training: text branch (MELD text) and audio branch (CREMA-D + RAVDESS, 8 kHz augmentation)."""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from callgraph.config import AppConfig
from callgraph.core.preprocess import simulate_telephone
from callgraph.data.labelmaps import shared_labels
from callgraph.data.manifest import read_manifest
from callgraph.utils.device import pick_device
from callgraph.utils.tracking import track_run


def out_dir(config: AppConfig, key: str) -> Path:
    return config.root / getattr(config.models, key)["output_dir"]


def load_audio_clip(config: AppConfig, rel: str, max_s: float, phone: bool) -> np.ndarray:
    """Load, resample to the target rate, optionally telephone-simulate, crop/pad to ``max_s``."""
    sr = int(config.pipeline.audio["target_sample_rate"])  # type: ignore[attr-defined]
    y, _ = librosa.load(str(config.root / rel), sr=sr, mono=True)
    if phone:
        y = simulate_telephone(y, sr, int(config.pipeline.audio["telephone_sample_rate"]))  # type: ignore[attr-defined]
    n = int(max_s * sr)
    return np.pad(y, (0, max(0, n - len(y))))[:n]


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if n <= 0 or len(df) <= n else df.sample(n=n, random_state=seed)


def train_text(config: AppConfig) -> dict[str, float]:
    """Fine-tune the text emotion model on MELD utterances."""
    m = config.models.emotion_text  # type: ignore[attr-defined]
    labels = shared_labels(config, "emotions")
    l2i = {lab: i for i, lab in enumerate(labels)}
    man = read_manifest(config, "meld")
    seed = int(m["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    def frame(split: str) -> pd.DataFrame:
        s = man[man.split == split]
        return pd.DataFrame(
            {
                "text": [lab["text"] for lab in s.labels],
                "label": [lab["emotion"] for lab in s.labels],
            }
        )

    tr, va = _sample(frame("train"), int(m["max_train_samples"]), seed), frame("validation")
    dev = pick_device(str(m["device"]))
    tok = AutoTokenizer.from_pretrained(m["backbone"])
    model = AutoModelForSequenceClassification.from_pretrained(
        m["backbone"], num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(m["learning_rate"]))
    bs = int(m["batch_size"])
    with track_run(config, "emotion_text", dict(m)) as log:
        for ep in range(int(m["epochs"])):
            model.train()
            idx = rng.permutation(len(tr))
            for i in range(0, len(idx), bs):
                b = tr.iloc[idx[i : i + bs]]
                enc = tok(
                    list(b.text),
                    truncation=True,
                    max_length=int(m["max_length"]),
                    padding=True,
                    return_tensors="pt",
                ).to(dev)
                loss = model(**enc, labels=torch.tensor([l2i[x] for x in b.label], device=dev)).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
            model.eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(va), bs):
                    enc = tok(
                        list(va.text.iloc[i : i + bs]),
                        truncation=True,
                        max_length=int(m["max_length"]),
                        padding=True,
                        return_tensors="pt",
                    ).to(dev)
                    preds += [labels[j] for j in model(**enc).logits.argmax(-1).tolist()]
            res = {
                "accuracy": accuracy_score(va.label, preds),
                "weighted_f1": f1_score(va.label, preds, average="weighted"),
            }
            logger.info("text epoch {} val {}", ep, res)
            log({f"val_{k}": v for k, v in res.items()})
    out = out_dir(config, "emotion_text")
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "labels.json").write_text(json.dumps(labels))
    return res


def train_audio(config: AppConfig) -> dict[str, float]:
    """Frozen wav2vec2 embeddings (+ 8 kHz copies) -> class-balanced logistic-regression head."""
    import pickle

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from callgraph.emotion.audio_features import AudioEmbedder

    m = config.models.emotion_audio  # type: ignore[attr-defined]
    labels = shared_labels(config, "emotions")
    seed = int(m["seed"])
    man = pd.concat([read_manifest(config, d) for d in ("crema_d", "ravdess")])
    tr = _sample(man[man.split == "train"], int(m["max_train_samples"]), seed)
    va = man[man.split == "validation"]
    emb, bs, max_s = AudioEmbedder(config), int(m["batch_size"]), float(m["max_audio_s"])

    def featurize(rows: pd.DataFrame, phone: bool) -> np.ndarray:
        out = []
        for i in range(0, len(rows), bs):
            ch = rows.iloc[i : i + bs]
            out.append(
                emb.embed(
                    [load_audio_clip(config, r.audio_path, max_s, phone) for r in ch.itertuples()]
                )
            )
            if (i // bs) % int(config.tracking.log_every_steps) == 0:  # type: ignore[attr-defined]
                logger.info("embedded {}/{} (phone={})", i, len(rows), phone)
        return np.concatenate(out)

    xs, ys = [featurize(tr, False)], [lab["emotion"] for lab in tr.labels]
    if m["telephone_aug"]:
        xs.append(featurize(tr, True))
        ys = ys + ys
    head = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(m["head"]["C"]), max_iter=int(m["head"]["max_iter"]), class_weight="balanced"
        ),
    )
    head.fit(np.concatenate(xs), ys)
    pred = head.predict(featurize(va, False))
    yv = [lab["emotion"] for lab in va.labels]
    res = {
        "accuracy": accuracy_score(yv, pred),
        "weighted_f1": f1_score(yv, pred, average="weighted"),
    }
    logger.info("audio head val {}", res)
    out = out_dir(config, "emotion_audio")
    out.mkdir(parents=True, exist_ok=True)
    (out / "head.pkl").write_bytes(pickle.dumps(head))
    (out / "labels.json").write_text(json.dumps(labels))
    return res
