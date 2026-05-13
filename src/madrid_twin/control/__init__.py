"""Control layer — heterogeneous multi-agent reinforcement learning.

Responsibilities
----------------

Define the multi-agent environment that wraps a SUMO scenario, expose it
through the PettingZoo API, and implement the training pipeline for the
heterogeneous agent set: signal-control agents at intersections,
lane-manager agents on candidate edges, and VMS / detour-advisor agents
at variable message signs. Add robustness via domain randomization and
an RARL-style adversarial demand agent.

Contract
--------

Environments live behind a PettingZoo ``ParallelEnv`` interface so they
are swappable. Policies are trained with MAPPO (primary) and QMIX
(ablation) under centralized training, decentralized execution. Trained
policies are saved as artefacts and registered in MLflow.

Submodules (planned)
--------------------

- ``env``         PettingZoo env wrapping the SUMO scenario runner.
- ``agents``      Signal, lane and VMS agent observation/action spaces.
- ``rewards``     Delay, throughput, spillback, emissions, equity terms.
- ``train_mappo`` MAPPO training entry point.
- ``adversary``   RARL-style adversarial demand perturbation agent.
- ``baselines``   Fixed-time, max-pressure, single-agent DRL references.
"""

from __future__ import annotations

__all__: list[str] = []
