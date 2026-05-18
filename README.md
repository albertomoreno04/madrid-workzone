# MADTwin — Madrid Workzone Digital Twin

> Heterogeneous multi-agent reinforcement learning for robust temporary
> traffic management around urban workzones, trained inside a calibrated
> SUMO digital twin of Madrid.

MADTwin is a research codebase whose headline contribution is a coordinated
controller — signal timings, lane re-allocations, and variable-message-sign
detour advisories — trained as a heterogeneous multi-agent reinforcement
learning policy under domain randomization and adversarial demand
perturbation. The controller closes the loop on a SUMO microsimulation
calibrated against open Madrid traffic data, with short-horizon forecasts
provided by a physics-informed spatiotemporal graph neural network and
distribution-free uncertainty intervals from conformal prediction.

## Architecture

The codebase is sliced into four packages that map one-to-one to the layers
of the architecture:

| Package | Layer | What it owns |
|---|---|---|
| `madrid_twin.data`    | Data & ground truth | Open-data ingestion (Madrid Ayuntamiento real-time intensity feed), workzone permits, census shapes. |
| `madrid_twin.sim`     | Calibrated twin     | SUMO network build, parametric workzone module, W-SPSA OD calibration, OD-to-trips emitter, curriculum scheduler, TraCI scenario runner. |
| `madrid_twin.predict` | Predictive          | Road-graph features, classical forecaster baselines, LWR physics residual, conformal uncertainty intervals, PI-GraphWaveNet. |
| `madrid_twin.control` | Control             | Heterogeneous MARL: signal / lane / VMS agents; composite reward; domain randomization + RARL adversary; PettingZoo env; RLlib MAPPO trainer. |
| `madrid_twin.eval`    | Evaluation          | Baseline metrics, GEH calibration validator, travel-time RMSE, ablations, robustness stress tests. |

## What works today

Everything in this list is tested in CI (198 tests as of Phase 4).

**Evaluation layer.** `madrid_twin.eval.baseline` parses a SUMO run bundle
into a typed `BaselineMetrics` JSON (delay, throughput, p95 travel time,
spillback proxy). `madrid_twin.eval.validation` provides the GEH statistic
plus a pass/fail report against Wisconsin DOT defaults (GEH<5 on ≥85% of
detectors), and a travel-time RMSE helper. `madrid_twin.eval.tracking` is
a thin MLflow run helper with phase + git-commit tagging.

**Simulation layer.** `madrid_twin.sim.workzone` is a typed `Workzone`
dataclass with strict validation and a SUMO additional-file emitter
that closes affected lanes via `<closingLaneReroute>`.
`madrid_twin.sim.network` is an opinionated `netconvert` wrapper that
builds a clean SUMO `.net.xml` from OSM. `madrid_twin.sim.calibrate`
implements W-SPSA OD calibration (Tympakianaki 2015) as pure numpy with
per-entry weights and Spall step-size schedules. `madrid_twin.sim.demand`
emits SUMO `<routes>` XML from an OD matrix, with demand-multiplier and
adversarial-perturbation hooks. `madrid_twin.sim.curriculum` schedules
workzone severity across training (linear ramp + piecewise stages).
`madrid_twin.sim.runner` is a TraCI-driven `SUMOScenarioRunner` that
satisfies the `Scenario` protocol so the PettingZoo env consumes it
interchangeably with the mock backend.

**Data layer.** `madrid_twin.data.open_data` is a typed httpx client for
the Ayuntamiento real-time intensity feed
(`https://informo.madrid.es/informo/tmadrid/pm.xml`) with retry,
on-disk caching, and a tolerant XML parser. `madrid_twin.data.probe` is
a one-shot smoke probe (wired as `make probe` and a DVC stage).

**Predictive layer.** `madrid_twin.predict.features` parses a SUMO
`.net.xml` into a `RoadGraph` and builds binary, distance-weighted, or
travel-time-weighted adjacency matrices (DCRNN convention).
`madrid_twin.predict.baselines` defines the `Forecaster` protocol with
three classical references — `HistoricalAverage`, `NaiveLastValue`,
`AR1Forecaster`. `madrid_twin.predict.physics` implements the LWR
conservation residual as a pure-numpy reference and a torch-loss
companion. `madrid_twin.predict.conformal` is a split-conformal wrapper
with finite-sample coverage guarantees and per-(horizon, node)
calibration. `madrid_twin.predict.stgnn` locks in the ST-GNN factory
interface; `madrid_twin.predict.stgnn_pignn` is the PI-GraphWaveNet
`nn.Module` itself (dilated temporal conv + diffusion graph conv +
LWR residual loss term, torch lazy-imported).

**Control layer.** `madrid_twin.control.rewards` builds a five-term
composite reward (negative delay + throughput − spillback − emissions
− Gini-equity across census zones) with configurable weights.
`madrid_twin.control.domain_randomization` samples capacity-drop,
demand multiplier, sensor noise and incident occurrence per episode.
`madrid_twin.control.adversary` projects raw RARL perturbations into a
two-stage feasible set (per-element clip + global L1 budget).
`madrid_twin.control.baselines` provides `FixedTimeController` and
`MaxPressureController` (Varaiya 2013). `madrid_twin.control.spaces`
exposes Gymnasium space factories for the three heterogeneous agent
classes. `madrid_twin.control.env` defines the `Scenario` protocol
and a `build_parallel_env` factory that wraps any compliant backend
in a PettingZoo `ParallelEnv`. `madrid_twin.control.train_mappo`
drives RLlib's PPO algorithm with per-class parameter sharing,
curriculum, DR and the adversary.

## Repository layout

```
madrid-workzone/
├── README.md                    # You are here
├── Makefile                     # Pinned, reproducible targets
├── pyproject.toml               # Package + dependency groups
├── .python-version              # Python 3.11.8 (pyenv-compatible)
├── .github/workflows/ci.yml     # Ruff + mypy + pytest on every PR
├── .dvc/                        # DVC config (data versioning)
├── dvc.yaml                     # DVC pipeline stages
├── data/                        # raw / interim / processed / external / outputs
├── scripts/
│   ├── analyze_baseline.py      # SUMO outputs -> BaselineMetrics JSON
│   ├── probe_open_data.py       # Madrid open-data smoke probe
│   ├── fit_baselines.py         # Phase 2: classical forecaster baselines + conformal
│   ├── smoke_control.py         # Phase 3: roll out fixed-time + max-pressure on the mock env
│   └── calibrate_od.py          # Phase 4: W-SPSA OD calibration demo on synthetic data
├── src/madrid_twin/
│   ├── config.py                # Central paths, MLflow URI, SUMO version pin
│   ├── data/                    # open_data, probe
│   ├── sim/                     # workzone, network, calibrate, demand, curriculum, runner
│   ├── predict/                 # features, baselines, physics, conformal, stgnn, stgnn_pignn
│   ├── control/                 # rewards, baselines, domain_randomization, adversary, spaces, env, train_mappo
│   └── eval/                    # baseline, validation, tracking
└── tests/                       # pytest suite (198 tests)
```

## Quickstart

### 1. Python environment

The project requires **Python 3.11+** (pinned in `.python-version`).

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,eval]"
```

Verify:

```bash
python -m pytest -q          # should print "198 passed"
python -m ruff check src scripts tests
python -m mypy src
```

On Linux/macOS (or Git Bash on Windows) `make check` collapses all three.

### 2. Phase 1 — Madrid open-data feed

```bash
make probe
```

Fetches the Ayuntamiento real-time intensity XML, prints a structured
report, and saves a timestamped snapshot under
`data/raw/traffic_intensity/`.

### 3. Phase 2 — fit forecaster baselines on synthetic data

```bash
make fit-baselines
```

Fits Historical Average / Naive / AR(1), attaches conformal intervals
at 90% nominal coverage, evaluates on a held-out split and writes
`data/outputs/predict/baselines_report.json`.

### 4. Phase 3 — control-layer smoke

```bash
make smoke-control
```

Rolls out `FixedTimeController` and `MaxPressureController` for 200
steps against the deterministic `MockQueueScenario` under the composite
reward, writing `data/outputs/control/smoke_report.json`.

### 5. Phase 4 — W-SPSA OD calibration smoke

```bash
make calibrate-od
```

Runs 500 iterations of W-SPSA on a 16-OD / 8-detector synthetic problem
with known ground truth. Writes `data/outputs/sim/calibrate_od_report.json`
with initial vs final loss and the L2 OD-error improvement.

### 6. SUMO 1.20.0 (only needed for the real simulator)

Pinned in `madrid_twin.config.SUMO_VERSION`. Install natively:

- **Ubuntu/Debian:** `sudo add-apt-repository ppa:sumo/stable && sudo apt-get install sumo=1.20.0* sumo-tools=1.20.0*`. Then `export SUMO_HOME=/usr/share/sumo`.
- **macOS:** `brew tap dlr-ts/sumo && brew install sumo`. Then `export SUMO_HOME=/opt/homebrew/opt/sumo/share/sumo`.
- **Windows:** download the 1.20.0 installer from <https://eclipse.dev/sumo/>, set `SUMO_HOME`, add `%SUMO_HOME%\bin` and `%SUMO_HOME%\tools` to PATH.

Then `pip install -e ".[sim]"` for the Python bindings.

### 7. Heavier extras

```bash
pip install -e ".[predict]"   # torch + torch-geometric-temporal + MAPIE  (PI-GraphWaveNet training)
pip install -e ".[control]"   # gymnasium + PettingZoo + RLlib            (MAPPO training)
pip install -e ".[data]"      # geopandas + shapely                       (zone-level equity analysis)
```

### 8. Experiment tracking

```bash
make mlflow-ui     # local MLflow UI on http://localhost:5000
```

## Reproducibility expectations

- **Pinned Python**: 3.11+ via `.python-version` and `requires-python`.
- **Pinned SUMO**: 1.20.0 in `madrid_twin.config.SUMO_VERSION`.
- **Dependency pins**: bounded versions in `pyproject.toml`; heavy
  groups (`predict`, `control`) kept optional so CI stays lean.
- **Data versioning**: DVC tracks `data/raw`, `data/interim`,
  `data/processed`. Outputs land in `data/outputs/`.
- **Experiment tracking**: MLflow, SQLite-backed locally; runs tagged
  with phase + git commit via `madrid_twin.eval.tracking.start_run`.
- **Quality gates**: Ruff (lint + format), mypy, pytest in CI.

## Status

Phases 0–4 are complete: scaffolding, calibration validators + open-data
ingestion + parametric workzone + OSM-to-SUMO build, predictive layer
(classical baselines + physics-informed graph net interface + conformal),
control layer (composite reward + DR + RARL adversary + heterogeneous
MARL env + MAPPO trainer + max-pressure baseline), and the SUMO TraCI
runner + W-SPSA OD calibration + curriculum scheduler that close the
loop. What remains is the experimental programme: calibrate the twin
against real Madrid data, train PI-GraphWaveNet on the resulting
trajectories, and run MAPPO end-to-end with curriculum + DR + adversary
turned on. Every step consumes a function or script that already ships.
