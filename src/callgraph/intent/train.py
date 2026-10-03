"""M2 training: TF-IDF+LR baseline and fine-tuned transformer (config-driven)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from callgraph.config import AppConfig
from callgraph.data.labelmaps import shared_labels
from callgraph.intent.data import training_examples
from callgraph.utils.device import pick_device
from callgraph.utils.tracking import track_run


def model_dir(config: AppConfig) -> Path:
    return config.root / config.models.intent["output_dir"]  # type: ignore[attr-defined]


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if n <= 0 or len(df) <= n else df.sample(n=n, random_state=seed)


def train_baseline(config: AppConfig) -> dict[str, Any]:
    """Fit TF-IDF + logistic regression; save to the intent model dir."""
    m = config.models.intent  # type: ignore[attr-defined]
    tr = _subsample(training_examples(config, "train"), int(m["max_train_samples"]), int(m["seed"]))
    te = training_examples(config, "test")
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=1000)
    )
    pipe.fit(tr.text, tr.label)
    pred = pipe.predict(te.text)
    metrics = {
        "accuracy": accuracy_score(te.label, pred),
        "macro_f1": f1_score(te.label, pred, average="macro"),
    }
    out = model_dir(config)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.pkl").write_bytes(pickle.dumps(pipe))
    logger.info("Baseline intent: {}", metrics)
    return metrics


def _batches(df: pd.DataFrame, bs: int, rng: np.random.Generator):
    idx = rng.permutation(len(df))
    for i in range(0, len(idx), bs):
        yield df.iloc[idx[i : i + bs]]


def train_transformer(config: AppConfig) -> dict[str, Any]:
    """Fine-tune the configured backbone; save weights + label list."""
    m = config.models.intent  # type: ignore[attr-defined]
    labels = shared_labels(config, "intents")
    l2i = {lab: i for i, lab in enumerate(labels)}
    seed = int(m["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = pick_device(str(m["device"]))
    tok = AutoTokenizer.from_pretrained(m["backbone"])
    model = AutoModelForSequenceClassification.from_pretrained(
        m["backbone"], num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i
    ).to(dev)
    tr = _subsample(training_examples(config, "train"), int(m["max_train_samples"]), seed)
    va = training_examples(config, "validation")
    opt = torch.optim.AdamW(model.parameters(), lr=float(m["learning_rate"]))
    with track_run(config, "intent_transformer", dict(m)) as log:
        for epoch in range(int(m["epochs"])):
            model.train()
            for step, b in enumerate(_batches(tr, int(m["batch_size"]), rng)):
                enc = tok(
                    list(b.text),
                    truncation=True,
                    max_length=int(m["max_length"]),
                    padding=True,
                    return_tensors="pt",
                ).to(dev)
                y = torch.tensor([l2i[x] for x in b.label], device=dev)
                loss = model(**enc, labels=y).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                if step % int(config.tracking.log_every_steps) == 0:
                    logger.info("epoch {} step {} loss {:.4f}", epoch, step, loss.item())
            res = evaluate_frame(
                model, tok, va, labels, dev, int(m["max_length"]), int(m["batch_size"])
            )
            logger.info("epoch {} val {}", epoch, res)
            log({f"val_{k}": v for k, v in res.items()})
    out = model_dir(config) / "transformer"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "labels.json").write_text(json.dumps(labels))
    return res


@torch.no_grad()
def evaluate_frame(
    model, tok, df: pd.DataFrame, labels: list[str], dev, max_len: int, bs: int
) -> dict[str, float]:
    """Accuracy and macro-F1 of a model on a (text, label) frame."""
    model.eval()
    preds: list[str] = []
    for i in range(0, len(df), bs):
        enc = tok(
            list(df.text.iloc[i : i + bs]),
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(dev)
        preds += [labels[j] for j in model(**enc).logits.argmax(-1).tolist()]
    return {
        "accuracy": accuracy_score(df.label, preds),
        "macro_f1": f1_score(df.label, preds, average="macro"),
    }
