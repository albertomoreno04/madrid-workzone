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
| `madrid_twin.predict` | Predictive          | ST-GNN baselines + physics-informed variant + conformal uncertainty intervals. |
| `madrid_twin.control` | Control             | Heterogeneous MARL: signal, lane and VMS agents; robustness via domain randomization + RARL. |
| `madrid_twin.eval`    | Evaluation          | Baseline metrics, GEH calibration validator, travel-time RMSE, ablations, robustness stress tests. |

## What works today

The current surface, with everything tested in CI:

- **`madrid_twin.eval.baseline`** — parse a SUMO run bundle (`summary.xml`,
  `tripinfo.xml`, `statistics.xml`) into a typed `BaselineMetrics` JSON
  (delay, throughput, p95 travel time, spillback proxy).
- **`madrid_twin.eval.validation`** — GEH statistic for calibration
  validation, batch GEH, pass/fail report against the Wisconsin DOT
  default thresholds (GEH<5 on ≥85% of detectors), plus travel-time RMSE.
- **`madrid_twin.eval.tracking`** — thin MLflow run helper with consistent
  phase + git-commit tagging. Lazy mlflow import keeps the package usable
  without the dev extras.
- **`madrid_twin.sim.workzone`** — a typed `Workzone` dataclass (id, lane
  closures, schedule, capacity-drop %), with serialization and a SUMO
  additional-file emitter that closes the affected lanes via
  `<closingLaneReroute>`.
- **`madrid_twin.sim.network`** — opinionated `netconvert` wrapper that
  builds a clean SUMO `.net.xml` from OSM with sensible defaults
  (geometry simplification, TLS import, junction joining, ramp guessing).
- **`madrid_twin.data.open_data`** — typed httpx client for the
  Ayuntamiento real-time intensity feed
  (`https://informo.madrid.es/informo/tmadrid/pm.xml`): fetch, retry with
  exponential backoff, optional on-disk cache, tolerant XML parser that
  copes with Spanish decimals and missing fields.
- **`madrid_twin.data.probe`** — one-shot smoke probe that hits the feed,
  reports response time and payload size, samples three detector readings,
  and writes a timestamped XML snapshot to `data/raw/traffic_intensity/`.

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
├── data/
│   ├── external/osm/            # OSM extract of the Madrid corridor
│   ├── raw/                     # DVC-tracked raw open-data pulls
│   ├── interim/                 # DVC-tracked intermediate artefacts
│   ├── processed/               # DVC-tracked processed inputs
│   └── outputs/                 # SUMO run outputs and metrics
├── scripts/
│   ├── analyze_baseline.py      # CLI: compute baseline metrics from SUMO outputs
│   └── probe_open_data.py       # CLI: smoke-probe the Madrid open-data feed
├── src/madrid_twin/
│   ├── __init__.py
│   ├── config.py                # Central paths, MLflow URI, SUMO version pin
│   ├── data/
│   │   ├── open_data.py         # Madrid Ayuntamiento traffic intensity client
│   │   └── probe.py             # Open-data smoke probe
│   ├── sim/
│   │   ├── network.py           # netconvert wrapper (OSM -> SUMO net.xml)
│   │   └── workzone.py          # Parametric workzone descriptor
│   ├── predict/                 # ST-GNN + conformal uncertainty (planned)
│   ├── control/                 # MARL controller (planned)
│   └── eval/
│       ├── baseline.py          # SUMO output bundle -> BaselineMetrics
│       ├── validation.py        # GEH + travel-time RMSE
│       └── tracking.py          # MLflow run helper
└── tests/                       # pytest suite
```

## Quickstart

### 1. Python environment

The project requires **Python 3.11** (pinned in `.python-version`). If you
use `pyenv` it will pick this up automatically; otherwise install Python
3.11 from python.org or your system package manager.

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,eval]"
```

Verify the install:

```bash
python -m pytest -q          # should print N passed (currently 60+)
python -m ruff check src scripts tests
python -m mypy src
```

On Linux/macOS (or Git Bash on Windows) `make check` collapses all three.

### 2. Hit the Madrid open-data feed (no SUMO needed)

```bash
make probe
```

That runs `scripts/probe_open_data.py`, which fetches the Ayuntamiento
real-time intensity XML, prints a structured report (URL, elapsed time,
payload size, detector counts), shows three sample readings, and saves a
timestamped snapshot under `data/raw/traffic_intensity/`.

### 3. SUMO 1.20.0 (only needed for simulation work)

The codebase pins SUMO **1.20.0** (`madrid_twin.config.SUMO_VERSION`).
Install it natively on your platform:

**Ubuntu / Debian.** Use the official Eclipse SUMO PPA:

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo=1.20.0* sumo-tools=1.20.0* sumo-doc=1.20.0*
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
```

**macOS.** Via Homebrew (verify the version):

```bash
brew tap dlr-ts/sumo
brew install sumo
echo 'export SUMO_HOME=/opt/homebrew/opt/sumo/share/sumo' >> ~/.zshrc
```

**Windows.** Download the official 1.20.0 installer from
<https://eclipse.dev/sumo/>, run it, and add the install dir to
`SUMO_HOME` plus `%SUMO_HOME%\bin` and `%SUMO_HOME%\tools` to `PATH`.

Then install the Python bindings:

```bash
python -m pip install -e ".[sim]"
```

> Note: `libsumo` (the fast in-process binding) is gated to non-Windows in
> `pyproject.toml` because upstream does not ship Windows wheels. On Windows
> you get `traci` + `sumolib`, which are sufficient for everything here.

### 4. Heavier extras (install only when you need them)

```bash
pip install -e ".[predict]"   # torch + torch-geometric-temporal + MAPIE
pip install -e ".[control]"   # gymnasium + PettingZoo + RLlib
pip install -e ".[data]"      # geopandas + shapely (zone analysis)
```

### 5. Experiment tracking (MLflow, local, no services)

The default tracking URI is a project-local SQLite file
(`sqlite:///mlruns.db`), with artefacts under `./mlartifacts/`.

```bash
make mlflow-ui     # launches MLflow UI on http://localhost:5000
```

Override the location via `MLFLOW_TRACKING_URI` / `MLFLOW_ARTIFACT_ROOT`
env vars — `madrid_twin.config` honours them.

## Reproducibility expectations

- **Pinned Python**: 3.11 via `.python-version` and the `requires-python`
  field in `pyproject.toml`.
- **Pinned SUMO**: 1.20.0, declared in `madrid_twin.config.SUMO_VERSION`
  and documented in the install steps above.
- **Dependency pins**: bounded versions in `pyproject.toml`. Heavy
  optional groups (`predict`, `control`) keep CI lean.
- **Data versioning**: DVC tracks `data/raw`, `data/interim`,
  `data/processed`. Outputs live in `data/outputs/`. Pipeline stages in
  `dvc.yaml`.
- **Experiment tracking**: MLflow, SQLite-backed locally. Runs tagged
  with phase + git commit via `madrid_twin.eval.tracking.start_run`.
- **Quality gates**: Ruff (lint + format), mypy, pytest, enforced in CI.
- **Pre-commit hooks**: `make precommit` installs them.

## Status

Repository scaffolding and the first calibration-validation tooling
(GEH, parametric workzones, Madrid open-data ingestion, OSM-to-SUMO
network build) are in place. Next up: W-SPSA demand calibration on the
expanded M-30 network, and the SUMO scenario runner that closes the loop
between control actions and detector observations.
