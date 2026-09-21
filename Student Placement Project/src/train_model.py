"""Training entrypoint wrapper for the Airflow DAG."""

from __future__ import annotations


def train() -> None:
    """Run model training using the project training module."""
    from src.models.train_model import train as _train

    _train()
