"""M6 training: fine-tune BART-base / Flan-T5 on DialogSum (config-driven, small subset on CPU)."""

from __future__ import annotations

import json

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from callgraph.config import AppConfig
from callgraph.data.manifest import read_manifest
from callgraph.utils.device import pick_device
from callgraph.utils.tracking import track_run


def dialog_text(config: AppConfig, path: str) -> str:
    doc = json.loads((config.root / path).read_text())
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in doc["turns"])


def train(config: AppConfig) -> None:
    """Fine-tune on DialogSum train; save to ``models.summarization.output_dir``."""
    m = config.models.summarization  # type: ignore[attr-defined]
    seed = int(m["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    man = read_manifest(config, "dialogsum")
    tr = man[man.split == "train"]
    tr = tr.sample(n=min(int(m["max_train_samples"]), len(tr)), random_state=seed)
    dev = pick_device(str(m["device"]))
    tok = AutoTokenizer.from_pretrained(m["seq2seq_backbone"])
    model = AutoModelForSeq2SeqLM.from_pretrained(m["seq2seq_backbone"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(m["learning_rate"]))
    bs = int(m["batch_size"])
    src = [dialog_text(config, p) for p in tr.transcript_path]
    tgt = [lab["summary"] for lab in tr.labels]
    with track_run(config, "summarization", dict(m)) as log:
        for ep in range(int(m["epochs"])):
            model.train()
            idx = rng.permutation(len(src))
            for k, i in enumerate(range(0, len(idx), bs)):
                b = idx[i : i + bs]
                enc = tok(
                    [src[j] for j in b],
                    truncation=True,
                    max_length=int(m["max_input_length"]),
                    padding=True,
                    return_tensors="pt",
                ).to(dev)
                lab = tok(
                    text_target=[tgt[j] for j in b],
                    truncation=True,
                    max_length=int(m["max_target_length"]),
                    padding=True,
                    return_tensors="pt",
                ).input_ids.to(dev)
                lab[lab == tok.pad_token_id] = -100
                loss = model(**enc, labels=lab).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                if k % int(config.tracking.log_every_steps) == 0:  # type: ignore[attr-defined]
                    logger.info("summ epoch {} step {} loss {:.4f}", ep, k, loss.item())
                    log({"loss": loss.item()})
    out = config.root / m["output_dir"]
    model.save_pretrained(out)
    tok.save_pretrained(out)
