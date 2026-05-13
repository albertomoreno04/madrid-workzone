# MADTwin — Madrid Workzone Digital Twin

> Heterogeneous multi-agent reinforcement learning for robust temporary
> traffic management around urban workzones, trained inside a calibrated
> SUMO digital twin of Madrid.

MADTwin is a research codebase whose headline contribution is a coordinated
controller — signal timings, lane re-allocations, and variable message sign
detour advisories — trained as a heterogeneous multi-agent reinforcement
learning (MARL) policy under domain randomization and adversarial demand
perturbation. The controller closes the loop on a SUMO microsimulation
calibrated against open Madrid traffic data, with short-horizon forecasts
provided by a physics-informed spatiotemporal graph neural network and
distribution-free uncertainty intervals from conformal prediction.

The full research plan, with phased roadmap, evaluation programme and
positioning against the literature, lives in
[`PROJECT_PLAN.md`](./PROJECT_PLAN.md).

## Architecture

The codebase is sliced into four packages that map one-to-one to the layers
of the architecture:

| Package | Layer | What it owns |
|---|---|---|
| `madrid_twin.data`    | Data & ground truth | Open-data ingestion, OD seeds, workzone descriptors, census shapes. |
| `madrid_twin.sim`     | Calibrated twin     | SUMO network build, W-SPSA demand calibration, parametric workzone module, TraCI bridge. |
| `madrid_twin.predict` | Predictive          | ST-GNN baselines + physics-informed variant + conformal uncertainty. |
| `madrid_twin.control` | Control             | Heterogeneous MARL: signal, lane and VMS agents; robustness via domain randomization + RARL. |
| `madrid_twin.eval`    | Evaluation          | Metrics, ablations, robustness stress tests, equity, statistical analysis. |

## Repository layout

```
madrid-workzone/
├── PROJECT_PLAN.md              # Full research plan
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
│   └── analyze_baseline.py      # CLI wrapper around eval.baseline
├── src/madrid_twin/
│   ├── __init__.py
│   ├── config.py                # Central paths, MLflow URI, SUMO version
│   ├── data/                    # ingest / probe (planned modules)
│   ├── sim/                     # network / demand / calibrate / workzone / runner
│   ├── predict/                 # baselines / pignn / conformal
│   ├── control/                 # env / agents / rewards / train_mappo / adversary
│   └── eval/
│       ├── baseline.py          # SUMO output bundle → BaselineMetrics
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
python -m pytest -q          # should print "26 passed"
python -m ruff check src scripts tests
python -m mypy src
```

On Linux/macOS (or Git Bash on Windows) you can collapse all three into
`make check`.

### 2. SUMO 1.20.0 (only needed for simulation work)

The codebase pins SUMO **1.20.0** (`madrid_twin.config.SUMO_VERSION`).
Install it natively on your platform:

**Ubuntu / Debian.** Use the official Eclipse SUMO PPA, which ships pinned
versions:

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo=1.20.0* sumo-tools=1.20.0* sumo-doc=1.20.0*
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
```

**macOS.** Easiest via Homebrew (binary may lag — verify the version):

```bash
brew tap dlr-ts/sumo
brew install sumo
echo 'export SUMO_HOME=/opt/homebrew/opt/sumo/share/sumo' >> ~/.zshrc
```

If brew does not have 1.20.0, build from source per the upstream docs.

**Windows.** Download the official 1.20.0 installer from
<https://eclipse.dev/sumo/> (the "1.20.0" Windows package), run it, and add
the install dir to `SUMO_HOME` plus `%SUMO_HOME%\bin` and
`%SUMO_HOME%\tools` to `PATH`.

Then install the Python bindings:

```bash
python -m pip install -e ".[sim]"
```

> Note: `libsumo` (the fast in-process binding) is gated to non-Windows in
> `pyproject.toml` because upstream does not ship Windows wheels. On Windows
> you get `traci` + `sumolib`, which are sufficient for everything in this
> repo.

### 3. Heavier extras (install only when you need them)

```bash
pip install -e ".[predict]"   # torch + torch-geometric-temporal + MAPIE
pip install -e ".[control]"   # gymnasium + PettingZoo + RLlib
pip install -e ".[data]"      # geopandas + shapely + http clients
```

### 4. Experiment tracking (MLflow, local, no services to run)

The default tracking URI is a project-local SQLite file
(`sqlite:///mlruns.db`), with artefacts written under `./mlartifacts/`.
Browse runs with:

```bash
make mlflow-ui     # starts MLflow UI on http://localhost:5000
```

Override the location by setting `MLFLOW_TRACKING_URI` /
`MLFLOW_ARTIFACT_ROOT` env vars — `madrid_twin.config` will pick them up.

## Reproducibility expectations

- **Pinned Python**: 3.11 via `.python-version` and the `requires-python`
  field in `pyproject.toml`.
- **Pinned SUMO**: 1.20.0, declared in `madrid_twin.config.SUMO_VERSION`
  and documented in the install steps above. Bumping is a deliberate act.
- **Dependency pins**: declared with bounded versions in `pyproject.toml`.
  Heavy optional groups (`predict`, `control`) keep CI lean.
- **Data versioning**: DVC tracks `data/raw`, `data/interim`,
  `data/processed`. Outputs live in `data/outputs/`.
- **Experiment tracking**: MLflow, SQLite-backed locally. Runs are tagged
  with phase + git commit via `madrid_twin.eval.tracking.start_run`.
- **Quality gates**: Ruff (lint + format), mypy, pytest, enforced in CI.
- **Pre-commit hooks**: `make precommit` installs them.

## Status

Phase 0 (scaffolding) is complete. Phase 1 (calibrated digital twin)
begins with extending the OSM corridor and writing the open-data probe.
See `PROJECT_PLAN.md` for the phased roadmap.
