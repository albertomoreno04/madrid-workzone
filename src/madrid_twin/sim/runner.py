"""SUMO TraCI scenario runner.

The closing piece of Phase 4: ``SUMOScenarioRunner`` satisfies the same
informal protocol that :class:`madrid_twin.control.env.MockQueueScenario`
already satisfies — ``reset() / step() / intersection_states() / close()`` —
so the PettingZoo env in :mod:`madrid_twin.control.env` plugs into a real
SUMO simulation with zero changes to the control layer.

``traci`` is lazy-imported, so this module is inspectable without the
``[sim]`` extras installed.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from madrid_twin.control.baselines import IntersectionState
from madrid_twin.control.rewards import NetworkState
from madrid_twin.sim.workzone import Workzone


@dataclass
class SUMOScenarioRunner:
    """TraCI-driven scenario runner for the real Madrid network."""

    sumo_cfg_path: Path
    workzone: Workzone | None = None
    step_s: float = 1.0
    gui: bool = False
    seed: int = 0
    traci_module: Any = None

    _started: bool = field(default=False, init=False)
    _step_count: int = field(default=0, init=False)
    _intersection_ids: tuple[str, ...] = field(default=(), init=False)

    @property
    def n_intersections(self) -> int:
        return len(self._intersection_ids)

    def _ensure_traci(self) -> Any:
        if self.traci_module is not None:
            return self.traci_module
        try:
            import traci
        except ImportError as exc:
            raise ImportError(
                "SUMOScenarioRunner requires the [sim] extras. Install with:\n"
                '    pip install -e ".[sim]"'
            ) from exc
        self.traci_module = traci
        return traci

    def reset(self) -> None:
        traci = self._ensure_traci()

        if self._started:
            traci.close()
            self._started = False

        binary = "sumo-gui" if self.gui else "sumo"
        cmd = [
            binary,
            "--configuration-file",
            str(self.sumo_cfg_path),
            "--seed",
            str(self.seed),
            "--step-length",
            str(self.step_s),
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
        ]
        if self.workzone is not None:
            wz_path = Path(self.sumo_cfg_path).parent / f"_wz_{self.workzone.id}.add.xml"
            self.workzone.write_sumo_additional(wz_path)
            cmd.extend(["--additional-files", str(wz_path)])

        traci.start(cmd)
        self._started = True
        self._step_count = 0
        try:
            self._intersection_ids = tuple(traci.trafficlight.getIDList())
        except Exception:
            self._intersection_ids = ()

    def step(self, phase_per_intersection: list[int]) -> NetworkState:
        if not self._started:
            raise RuntimeError("SUMOScenarioRunner.reset() must be called first")
        traci = self.traci_module

        for tls_id, phase in zip(self._intersection_ids, phase_per_intersection, strict=False):
            with contextlib.suppress(Exception):
                traci.trafficlight.setPhase(tls_id, int(phase))

        traci.simulationStep()
        self._step_count += 1

        return self._build_network_state()

    def _build_network_state(self) -> NetworkState:
        traci = self.traci_module
        vehicle_ids: list[str] = []
        try:
            vehicle_ids = list(traci.vehicle.getIDList())
        except Exception:
            vehicle_ids = []

        delays_s = np.array(
            [float(traci.vehicle.getAccumulatedWaitingTime(vid)) for vid in vehicle_ids]
            if vehicle_ids
            else [],
            dtype=np.float64,
        )

        try:
            throughput = int(traci.simulation.getArrivedNumber())
        except Exception:
            throughput = 0

        try:
            edge_ids = tuple(traci.edge.getIDList())
            edge_ids = tuple(e for e in edge_ids if not e.startswith(":"))
            queue_lengths_m = np.array(
                [float(traci.edge.getLastStepHaltingNumber(e)) * 7.5 for e in edge_ids]
            )
            link_lengths_m = np.array([float(traci.lane.getLength(f"{e}_0")) for e in edge_ids])
            link_lengths_m = np.maximum(link_lengths_m, 1.0)
        except Exception:
            queue_lengths_m = np.array([1.0])
            link_lengths_m = np.array([100.0])

        try:
            emissions_g = sum(float(traci.edge.getCO2Emission(e)) for e in edge_ids) / 1000.0
        except Exception:
            emissions_g = 0.0

        per_zone_impact = np.zeros(0, dtype=np.float64)

        return NetworkState(
            delays_s=delays_s,
            throughput_veh=throughput,
            queue_lengths_m=queue_lengths_m,
            link_lengths_m=link_lengths_m,
            emissions_g=emissions_g,
            travel_time_impact_per_zone_s=per_zone_impact,
        )

    def intersection_states(self) -> list[IntersectionState]:
        if not self._started:
            raise RuntimeError("SUMOScenarioRunner.reset() must be called first")
        traci = self.traci_module
        out: list[IntersectionState] = []
        for tls_id in self._intersection_ids:
            try:
                lanes_in = traci.trafficlight.getControlledLanes(tls_id)
                n_phases = len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
                inc = np.array([float(traci.lane.getLastStepHaltingNumber(ln)) for ln in lanes_in])
                if inc.size < n_phases:
                    inc = np.concatenate([inc, np.zeros(n_phases - inc.size)])
                else:
                    inc = inc[:n_phases]
                out.append(
                    IntersectionState(
                        intersection_id=tls_id,
                        n_phases=n_phases,
                        incoming_queues=inc,
                        outgoing_queues=np.zeros(n_phases),
                    )
                )
            except Exception:
                continue
        return out

    def close(self) -> None:
        if self._started and self.traci_module is not None:
            with contextlib.suppress(Exception):
                self.traci_module.close()
            self._started = False


__all__ = ["SUMOScenarioRunner"]
