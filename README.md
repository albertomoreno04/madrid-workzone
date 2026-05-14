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
| `madrid_twin.sim`     | Calibrated twin     | SUMO network build (`netconvert` wrapper), parametric workzone module, W-SPSA demand calibration, TraCI bridge. |
| `madrid_twin.predict` | Predictive          | Road-graph features, ST-GNN baselines + physics-informed variant, conformal uncertainty intervals. |
| `madrid_twin.control` | Control             | Heterogeneous MARL: signal, lane and VMS agents; robustness via domain randomization + RARL. |
| `madrid_twin.eval`    | Evaluation          | Baseline metrics, GEH calibration validator, travel-time RMSE, ablations, robustness stress tests. |

## What works today

Everything in this list is tested in CI.

**Evaluation layer.** `madrid_twin.eval.baseline` parses a SUMO run bundle
into a typed `BaselineMetrics` JSON (delay, throughput, p95 travel time,
spillback proxy). `madrid_twin.eval.validation` provides the GEH statistic
plus a pass/fail report against Wisconsin DOT defaults (GEH<5 on ≥85% of
detectors), and a travel-time RMSE helper. `madrid_twin.eval.tracking` is
a thin MLflow run helper with consistent phase + git-commit tagging.

**Simulation layer.** `madrid_twin.sim.workzone` is a typed `Workzone`
dataclass (id, lane closures, schedule, capacity-drop %) with
serialization and a SUMO additional-file emitter that closes affected
lanes via `<closingLaneReroute>`. `madrid_twin.sim.network` is an
opinionated `netconvert` wrapper that builds a clean SUMO `.net.xml`
from OSM with sensible defaults.

**Data layer.** `madrid_twin.data.open_data` is a typed httpx client for
the Ayuntamiento real-time intensity feed
(`https://informo.madrid.es/informo/tmadrid/pm.xml`) with retry,
on-disk caching, and a tolerant XML parser. `madrid_twin.data.probe` is
a one-shot smoke probe (also wired as the `make probe` target and the
`probe_traffic_intensity` DVC stage).

**Predictive layer.** `madrid_twin.predict.features` parses a SUMO
`.net.xml` into a `RoadGraph` and builds binary, distance-weighted, or
travel-time-weighted adjacency matrices (the DCRNN convention).
`madrid_twin.predict.baselines` defines the `Forecaster` protocol and a
`ForecastBundle` return type, with three classical baselines —
`HistoricalAverage`, `NaiveLastValue`, and `AR1Forecaster` — that
together cover the non-deep references every ST-GNN paper compares
against. `madrid_twin.predict.physics` implements the LWR conservation
residual as a pure-numpy reference (it will be wrapped as a torch loss
in Phase 3). `madrid_twin.predict.conformal` is a split-conformal
wrapper that turns any `Forecaster` into one with coverage-guaranteed
intervals — supports per-(horizon, node) calibration to handle the
strong heteroskedasticity of urban traffic. `madrid_twin.predict.stgnn`
locks in the ST-GNN interface (`STGNNConfig`, `load_stgnn`) with
torch as a lazy import — the actual DCRNN / GraphWaveNet / PI-GraphWaveNet
classes land in Phase 3 against real trajectories.

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
│   └── fit_baselines.py         # Fit Phase 2 forecaster baselines, report MAE + conformal coverage
├── src/madrid_twin/
│   ├── config.py                # Central paths, MLflow URI, SUMO version pin
│   ├── data/                    # open_data, probe
│   ├── sim/                     # network, workzone
│   ├── predict/                 # features, baselines, physics, conformal, stgnn
│   ├── control/                 # MARL controller (Phase 3)
│   └── eval/                    # baseline, validation, tracking
└── tests/                       # pytest suite (119 tests as of Phase 2)
```

## Quickstart

### 1. Python environment

The project requires **Python 3.11** (pinned in `.python-version`).

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,eval]"
```

Verify:

```bash
python -m pytest -q          # should print "119 passed"
python -m ruff check src scripts tests
python -m mypy src
```

On Linux/macOS (or Git Bash on Windows) `make check` collapses all three.

### 2. Phase 2 entry point: fit the forecaster baselines on synthetic data

```bash
make fit-baselines
```

Runs `scripts/fit_baselines.py`, fits Historical Average / Naive / AR(1),
attaches conformal intervals at 90% nominal coverage, evaluates on a
held-out split and writes `data/outputs/predict/baselines_report.json`.
Useful as a smoke check and as a reference call site for the real
calibration loop that will land once Phase 1 emits trajectories.

### 3. Phase 1 entry point: hit the Madrid open-data feed

```bash
make probe
```

Fetches the Ayuntamiento real-time intensity XML, prints a structured
report, and saves a timestamped snapshot under
`data/raw/traffic_intensity/`.

### 4. SUMO 1.20.0 (only needed for simulation work)

Pinned in `madrid_twin.config.SUMO_VERSION`. Install natively:

- **Ubuntu/Debian:** `sudo add-apt-repository ppa:sumo/stable && sudo apt-get install sumo=1.20.0* sumo-tools=1.20.0*`. Then `export SUMO_HOME=/usr/share/sumo`.
- **macOS:** `brew tap dlr-ts/sumo && brew install sumo`. Then `export SUMO_HOME=/opt/homebrew/opt/sumo/share/sumo`.
- **Windows:** download the 1.20.0 installer from <https://eclipse.dev/sumo/>, set `SUMO_HOME`, add `%SUMO_HOME%\bin` and `%SUMO_HOME%\tools` to PATH.

Then `pip install -e ".[sim]"` for the Python bindings.

### 5. Heavier extras

```bash
pip install -e ".[predict]"   # torch + torch-geometric-temporal + MAPIE  (for Phase 3 ST-GNN models)
pip install -e ".[control]"   # gymnasium + PettingZoo + RLlib            (for Phase 3 MARL training)
pip install -e ".[data]"      # geopandas + shapely                       (zone-level equity analysis)
```

### 6. Experiment tracking

```bash
make mlflow-ui     # local MLflow UI on http://localhost:5000
```

Default tracking URI: `sqlite:///mlruns.db`. Override via
`MLFLOW_TRACKING_URI` / `MLFLOW_ARTIFACT_ROOT` env vars
(`madrid_twin.config` honours them).

## Reproducibility expectations

- **Pinned Python**: 3.11 via `.python-version` and `requires-python`.
- **Pinned SUMO**: 1.20.0 in `madrid_twin.config.SUMO_VERSION`.
- **Dependency pins**: bounded versions in `pyproject.toml`; heavy
  groups (`predict`, `control`) kept optional so CI stays lean.
- **Data versioning**: DVC tracks `data/raw`, `data/interim`,
  `data/processed`. Outputs land in `data/outputs/`.
- **Experiment tracking**: MLflow, SQLite-backed locally; runs tagged
  with phase + git commit via `madrid_twin.eval.tracking.start_run`.
- **Quality gates**: Ruff (lint + format), mypy, pytest enforced in CI.

## Status

Phase 0 (scaffolding), Phase 1 (calibration validators + Madrid
open-data ingestion + parametric workzone + OSM-to-SUMO build) and
Phase 2 (road-graph features + classical forecaster baselines +
LWR conservation residual + split-conformal intervals + ST-GNN
interface skeleton) are complete. Next up: Phase 3 — heterogeneous
MARL controller training against the calibrated twin, with the
torch-backed ST-GNN models (DCRNN, GraphWaveNet, PI-GraphWaveNet)
plugging into the predict.stgnn factory.
