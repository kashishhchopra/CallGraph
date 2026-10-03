"""M2 inference: transformer (or baseline) intent classifier, early-intent and secondary intents."""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from callgraph.config import AppConfig
from callgraph.intent.data import customer_text
from callgraph.intent.train import model_dir
from callgraph.registry import MODULES, AnalysisModule
from callgraph.schemas import CallTranscript, IntentOutput, IntentScore
from callgraph.utils.device import pick_device

FULL = "full"


@lru_cache(maxsize=2)
def _load_transformer(path: str, dev: str) -> tuple[Any, Any, list[str]]:
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval()
    return tok, model, json.loads(open(f"{path}/labels.json").read())


class IntentClassifier:
    """Predict shared-label intent probabilities from customer text."""

    def __init__(self, config: AppConfig, kind: str = "transformer") -> None:
        self.config = config
        self.kind = kind
        base = model_dir(config)
        if kind == "transformer":
            self.dev = str(pick_device(str(config.models.intent["device"])))  # type: ignore[attr-defined]
            self.tok, self.model, self.labels = _load_transformer(
                str(base / "transformer"), self.dev
            )
        else:
            self.pipe = pickle.loads((base / "baseline.pkl").read_bytes())
            self.labels = list(self.pipe.classes_)

    @torch.no_grad()
    def proba(self, texts: list[str]) -> np.ndarray:
        """(n, labels) probability matrix."""
        if self.kind == "baseline":
            return self.pipe.predict_proba(texts)
        out = []
        bs = int(self.config.models.intent["batch_size"])  # type: ignore[attr-defined]
        for i in range(0, len(texts), bs):
            enc = self.tok(
                texts[i : i + bs],
                truncation=True,
                max_length=int(self.config.models.intent["max_length"]),  # type: ignore[attr-defined]
                padding=True,
                return_tensors="pt",
            ).to(self.dev)
            out.append(torch.softmax(self.model(**enc).logits, -1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, call: CallTranscript) -> IntentOutput:
        """Full-call intent plus the prediction available at each configured time cutoff."""
        th = self.config.thresholds.intent  # type: ignore[attr-defined]
        turns = [t.model_dump() for t in call.turns]
        cutoffs = list(self.config.evaluation.intent_cutoffs_s)  # type: ignore[attr-defined]
        texts = [customer_text(turns, until_s=float(c)) or " " for c in cutoffs] + [
            customer_text(turns) or " "
        ]
        probs = self.proba(texts)
        by_cutoff = {
            str(c): self.labels[int(p.argmax())] for c, p in zip(cutoffs, probs[:-1], strict=False)
        }
        full = probs[-1]
        order = np.argsort(-full)
        top_k = [
            IntentScore(name=self.labels[i], prob=float(full[i])) for i in order[: int(th["top_k"])]
        ]
        secondary = [s for s in top_k[1:] if s.prob >= float(th["secondary_threshold"])]
        first = next((t for t in call.turns if t.role == "customer"), None)
        return IntentOutput(
            call_id=call.call_id,
            top_intent=top_k[0].name,
            top_k=top_k,
            secondary=secondary,
            detected_at_s=first.end if first else None,
            by_cutoff=by_cutoff,
        )


@MODULES.register("intent")
class IntentModule(AnalysisModule):
    """Registered M2 plugin."""

    _clf: IntentClassifier | None = None

    def run(self, call: CallTranscript, config: AppConfig, **context: Any) -> IntentOutput:
        if self._clf is None:
            type(self)._clf = IntentClassifier(config)
        return self._clf.predict(call)  # type: ignore[union-attr]
