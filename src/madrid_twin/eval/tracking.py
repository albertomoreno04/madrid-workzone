"""Thin MLflow helper used across the project.

All MLflow access flows through ``start_run`` so we get consistent run tags
(git commit, scenario id, phase) without every caller having to remember.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from collections.abc import Iterator, Mapping
from typing import Any

from madrid_twin.config import MLFLOW_ARTIFACT_ROOT, MLFLOW_TRACKING_URI


def _git_commit() -> str | None:
    """Best-effort short commit hash; ``None`` if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _set_tracking() -> None:
    import mlflow  # local import: keeps the package optional at import time

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    MLFLOW_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def start_run(
    *,
    experiment: str,
    run_name: str | None = None,
    tags: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Open an MLflow run with consistent tagging.

    Yields the ``mlflow.ActiveRun`` so callers can ``log_metric`` etc.
    inside the ``with`` block. Imports MLflow lazily so the package itself
    is usable without the dev extras installed.
    """
    import mlflow

    _set_tracking()
    mlflow.set_experiment(experiment)

    merged_tags: dict[str, str] = {
        "phase": os.environ.get("MADTWIN_PHASE", "phase-0"),
    }
    commit = _git_commit()
    if commit:
        merged_tags["git_commit"] = commit
    if tags:
        merged_tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=merged_tags) as run:
        if params:
            mlflow.log_params(dict(params))
        yield run


__all__ = ["start_run"]
