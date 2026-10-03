"""Intent evaluation: accuracy / macro-F1 / confusion matrix, baseline vs transformer, early-intent curve (exp 3)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from callgraph.config import AppConfig
from callgraph.data.labelmaps import shared_labels
from callgraph.intent.data import customer_text, load_frame
from callgraph.intent.model import IntentClassifier
from eval.common import save_fig, save_result


def score(y: list[str], pred: list[str]) -> dict[str, float]:
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}


def run(config: AppConfig) -> dict[str, Any]:
    ev = config.evaluation  # type: ignore[attr-defined]
    labels = shared_labels(config, "intents")
    clfs = {
        "tfidf_logreg": IntentClassifier(config, "baseline"),
        "transformer": IntentClassifier(config, "transformer"),
    }
    rows: list[dict[str, Any]] = []
    for ds, split, n in (
        ("abcd", "test", ev.subsets["intent"]["abcd_test"]),
        ("banking77", "test", ev.subsets["intent"]["abcd_test"]),
        ("spokenwoz", "test", ev.subsets["intent"]["spokenwoz"]),
    ):
        df = load_frame(config, ds, [split]).head(int(n))
        texts = [customer_text(t) for t in df["turns"]]
        for name, clf in clfs.items():
            pred = [clf.labels[i] for i in clf.proba(texts).argmax(1)]
            rows.append(
                {"dataset": ds, "model": name, "n": len(df), **score(list(df["intent"]), pred)}
            )
            if ds == ev.intent["confusion_dataset"] and name == "transformer":  # type: ignore[index]
                cm = confusion_matrix(df["intent"], pred, labels=labels)
                pd.DataFrame(cm, index=labels, columns=labels).to_csv(
                    save_result(config, "intent_confusion", {"labels": labels}).with_suffix(".csv")
                )
                plt.figure(figsize=(7, 6))
                plt.imshow(cm, cmap="Blues")
                plt.xticks(range(len(labels)), labels, rotation=90)
                plt.yticks(range(len(labels)), labels)
                plt.title(f"Intent confusion ({ds} test, transformer)")
                save_fig(config, "intent_confusion")
    # Early intent: accuracy at time cutoffs on SpokenWOZ (timed ground-truth text) -- experiment 3
    curve = []
    if ev.experiments["early_intent_curve"]:
        df = load_frame(config, "spokenwoz", ["test"]).head(int(ev.subsets["intent"]["spokenwoz"]))
        cuts = sorted(set(list(ev.intent["curve_cutoffs_s"]) + list(ev.intent_cutoffs_s)))  # type: ignore[index]
        for name, clf in clfs.items():
            for c in [*cuts, None]:
                texts = [customer_text(t, until_s=c) or " " for t in df["turns"]]
                pred = [clf.labels[i] for i in clf.proba(texts).argmax(1)]
                curve.append(
                    {
                        "model": name,
                        "cutoff_s": "full" if c is None else c,
                        **score(list(df["intent"]), pred),
                    }
                )
        cdf = pd.DataFrame(curve)
        save_result(config, "exp3_early_intent_curve", {"rows": curve}, cdf)
        plt.figure(figsize=(6, 4))
        for name, g in cdf.groupby("model"):
            g = g[g.cutoff_s != "full"]
            plt.plot(g.cutoff_s.astype(float), g.accuracy, marker="o", label=name)
        plt.xlabel("seconds into call")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("Early intent accuracy (SpokenWOZ, domain-derived labels)")
        save_fig(config, "exp3_early_intent_curve")
    save_result(config, "intent_metrics", {"rows": rows}, pd.DataFrame(rows))
    return {"rows": rows, "curve": curve}
