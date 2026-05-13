"""Simulation layer — the SUMO-based calibrated digital twin.

Responsibilities
----------------

Build the SUMO network from OSM, generate and calibrate demand against
detector counts using W-SPSA, apply parametric workzones (lane closures
plus capacity-drop following HCM 2016 with empirical corrections), run
scenarios through TraCI/libsumo, and stream observations to the predict
and control layers.

Contract
--------

The simulator is treated as a pure function from
``(network, demand, workzone_descriptor, control_policy)`` to a stream of
observations and a final SUMO output bundle (``summary.xml``,
``tripinfo.xml``, ``statistics.xml``). Stochasticity is controlled by an
explicit seed argument so every experiment is reproducible.

Submodules (planned)
--------------------

- ``network``    OSM → SUMO via netconvert wrappers.
- ``demand``     Demand generation, OD seeds.
- ``calibrate``  W-SPSA calibration loop against detector counts.
- ``workzone``   Parametric workzone module (edges, lanes, schedule, drop).
- ``runner``     Scenario runner with TraCI/libsumo bridge.
- ``routing``    Stochastic UE / boundedly-rational rerouting on VMS.
"""

from __future__ import annotations

__all__: list[str] = []
