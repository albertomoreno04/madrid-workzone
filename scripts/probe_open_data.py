"""CLI wrapper around :mod:`madrid_twin.data.probe`.

Run from the repo root::

    python -m scripts.probe_open_data

or via the ``probe`` Makefile target. Hits the Madrid Ayuntamiento
real-time traffic intensity feed, prints a structured report, and
writes a timestamped XML snapshot under ``data/raw/traffic_intensity/``.
"""

from __future__ import annotations

from madrid_twin.data.probe import main

if __name__ == "__main__":
    main()
