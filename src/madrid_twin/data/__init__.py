"""Data layer — ingest, version and stage all inputs to the digital twin.

Responsibilities
----------------

This subpackage owns every byte that enters MADTwin from the outside world.
That includes the Madrid Ayuntamiento traffic intensity feed for M-30
detectors, the workzone permits portal, EMT's GTFS-RT for buses, AEMET
weather observations, DGT historical traffic counts, OSM extracts, and INE
census shapefiles for the equity analysis.

Contract
--------

All public functions in this subpackage write into the DVC-tracked
``data/`` hierarchy (see ``madrid_twin.config``) and return file paths,
never in-memory dataframes. Downstream layers (``sim``, ``predict``,
``eval``) load from those paths so that everything is reproducible from
DVC alone.

Submodules (planned)
--------------------

- ``open_data``  Polling clients for Ayuntamiento, EMT, AEMET, DGT.
- ``osm``        OSM extract handling and network preparation hand-off.
- ``zones``      INE census zone shapes for equity computation.
- ``probe``      The "open-data probe" smoke client used to confirm we can
                 pull every required feed at the cadence we need.
"""

from __future__ import annotations

__all__: list[str] = []
