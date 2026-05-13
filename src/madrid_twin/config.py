"""Central paths and constants for MADTwin.

All file-system locations the codebase reads from or writes to flow through
this module. The intent is that downstream code never hard-codes ``"data/..."``
or similar — that way redirecting outputs (e.g. for a CI run, or a different
DVC remote) only touches one place.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root is resolved by walking up from this file. The package layout
# is ``<root>/src/madrid_twin/config.py``, so root is three parents up.
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
SRC_ROOT: Path = PACKAGE_ROOT.parent
REPO_ROOT: Path = SRC_ROOT.parent


def _path_from_env(var: str, default: Path) -> Path:
    """Resolve ``default`` unless ``var`` overrides it. Always returns absolute."""
    raw = os.environ.get(var)
    if raw:
        return Path(raw).expanduser().resolve()
    return default.resolve()


# ---------------------------------------------------------------------------
# Data lake layout — mirrors a standard ``raw / interim / processed`` split.
# ---------------------------------------------------------------------------

DATA_DIR: Path = _path_from_env("MADTWIN_DATA_DIR", REPO_ROOT / "data")
DATA_RAW: Path = DATA_DIR / "raw"
DATA_INTERIM: Path = DATA_DIR / "interim"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DATA_EXTERNAL: Path = DATA_DIR / "external"
DATA_OUTPUTS: Path = DATA_DIR / "outputs"

# Baseline analysis outputs (kept compatible with the original scripts layout).
BASELINE_DIR: Path = DATA_OUTPUTS / "baseline"
BASELINE_SUMMARY_XML: Path = BASELINE_DIR / "summary.xml"
BASELINE_TRIPINFO_XML: Path = BASELINE_DIR / "tripinfo.xml"
BASELINE_STATISTICS_XML: Path = BASELINE_DIR / "statistics.xml"
BASELINE_METRICS_JSON: Path = BASELINE_DIR / "baseline_metrics.json"

# OSM corridor (already in the repo).
OSM_CORRIDOR: Path = DATA_EXTERNAL / "osm" / "madrid_corridor.osm"

# ---------------------------------------------------------------------------
# MLflow.
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{REPO_ROOT / 'mlruns.db'}",
)
MLFLOW_ARTIFACT_ROOT: Path = _path_from_env("MLFLOW_ARTIFACT_ROOT", REPO_ROOT / "mlartifacts")

# ---------------------------------------------------------------------------
# SUMO.
# ---------------------------------------------------------------------------

# Pinned SUMO version. Document of record for the SUMO release the project
# is calibrated against. Bumping this is a deliberate act — update the
# README install instructions and re-validate at the same time.
SUMO_VERSION: str = "1.20.0"

SUMO_HOME: Path | None = Path(os.environ["SUMO_HOME"]) if os.environ.get("SUMO_HOME") else None


__all__ = [
    "PACKAGE_ROOT",
    "SRC_ROOT",
    "REPO_ROOT",
    "DATA_DIR",
    "DATA_RAW",
    "DATA_INTERIM",
    "DATA_PROCESSED",
    "DATA_EXTERNAL",
    "DATA_OUTPUTS",
    "BASELINE_DIR",
    "BASELINE_SUMMARY_XML",
    "BASELINE_TRIPINFO_XML",
    "BASELINE_STATISTICS_XML",
    "BASELINE_METRICS_JSON",
    "OSM_CORRIDOR",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_ARTIFACT_ROOT",
    "SUMO_VERSION",
    "SUMO_HOME",
]
