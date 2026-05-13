"""Smoke tests for the package layout.

These tests confirm that the four-layer subpackage structure resolves and
that the top-level package exposes its declared API. They are intentionally
cheap so CI catches accidental import-time regressions.
"""

from __future__ import annotations

import importlib


def test_top_level_version() -> None:
    import madrid_twin

    assert isinstance(madrid_twin.__version__, str)
    assert madrid_twin.__version__.count(".") == 2


def test_subpackages_import_cleanly() -> None:
    for name in (
        "madrid_twin.data",
        "madrid_twin.sim",
        "madrid_twin.predict",
        "madrid_twin.control",
        "madrid_twin.eval",
    ):
        importlib.import_module(name)


def test_eval_reexports_baseline_api() -> None:
    from madrid_twin.eval import (
        BaselineMetrics,
        compute_baseline_metrics,
        parse_statistics,
        parse_summary,
        parse_tripinfo,
    )

    assert callable(compute_baseline_metrics)
    assert callable(parse_summary)
    assert callable(parse_tripinfo)
    assert callable(parse_statistics)
    assert BaselineMetrics.__module__ == "madrid_twin.eval.baseline"


def test_config_paths_resolve() -> None:
    from madrid_twin import config

    assert config.REPO_ROOT.is_dir()
    assert config.SRC_ROOT.is_dir()
    assert config.PACKAGE_ROOT.is_dir()
    assert config.SUMO_VERSION == "1.20.0"
