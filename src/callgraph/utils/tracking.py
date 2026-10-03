"""Experiment tracking (MLflow / W&B / none) selected in settings.yaml."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from loguru import logger

from callgraph.config import AppConfig


@contextmanager
def track_run(config: AppConfig, name: str, params: dict[str, Any]) -> Iterator[Any]:
    """Context manager yielding a ``log(metrics: dict)`` callable."""
    backend = str(config.tracking.backend)  # type: ignore[attr-defined]
    if backend == "mlflow":
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{config.path('outputs_dir') / 'mlflow.db'}")
        mlflow.set_experiment(str(config.tracking.experiment_name))  # type: ignore[attr-defined]
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {k: str(v)[: int(config.tracking.max_param_chars)] for k, v in params.items()}
            )
            yield lambda m: mlflow.log_metrics(
                {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
            )
    elif backend == "wandb":
        import wandb

        run = wandb.init(project=str(config.tracking.experiment_name), name=name, config=params)  # type: ignore[attr-defined]
        yield lambda m: run.log(m)
        run.finish()
    else:
        logger.info("Tracking disabled")
        yield lambda m: None
