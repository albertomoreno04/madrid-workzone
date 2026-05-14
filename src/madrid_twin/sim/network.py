"""SUMO network build from OpenStreetMap.

Thin, deterministic wrapper around SUMO's ``netconvert`` binary that
converts an OSM extract into a clean SUMO ``.net.xml``. The wrapper
encodes a small, opinionated set of defaults that match what the rest
of MADTwin assumes: traffic lights are imported from OSM, edges are
joined at near-coincident junctions, dead-end approaches that confuse
microsimulation are removed, and a road-class filter restricts the
output to the urban subset we care about (motorway through residential).

The actual binary call goes through :func:`subprocess.run` so we can
inject a custom runner in tests without ever shelling out.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Sensible OSM road classes for an urban traffic model. These are passed
# to netconvert via ``--keep-edges.by-vclass passenger`` (broader) combined
# with explicit type filters. We expose them as a constant so callers can
# tweak.
DEFAULT_ROAD_TYPES: tuple[str, ...] = (
    "highway.motorway",
    "highway.motorway_link",
    "highway.trunk",
    "highway.trunk_link",
    "highway.primary",
    "highway.primary_link",
    "highway.secondary",
    "highway.secondary_link",
    "highway.tertiary",
    "highway.tertiary_link",
    "highway.residential",
    "highway.unclassified",
)


@dataclass(frozen=True)
class NetconvertResult:
    """Captures the outcome of a netconvert run."""

    net_xml: Path
    command: tuple[str, ...]
    stdout: str
    stderr: str


class NetconvertError(RuntimeError):
    """Raised when netconvert returns a non-zero exit code."""

    def __init__(self, returncode: int, stderr: str, command: Sequence[str]) -> None:
        super().__init__(
            f"netconvert failed with exit code {returncode}\n"
            f"command: {' '.join(command)}\n"
            f"stderr (last 1KB):\n{stderr[-1024:]}"
        )
        self.returncode = returncode
        self.stderr = stderr
        self.command = tuple(command)


# Subprocess runner type. We accept any callable with the same shape as
# ``subprocess.run`` so tests can swap in a fake.
SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


def build_network_from_osm(
    osm_path: Path,
    output_net: Path,
    *,
    netconvert_binary: str | None = None,
    road_types: Sequence[str] = DEFAULT_ROAD_TYPES,
    extra_args: Sequence[str] = (),
    runner: SubprocessRunner = subprocess.run,
) -> NetconvertResult:
    """Convert an OSM file to a SUMO ``.net.xml`` via ``netconvert``.

    Parameters
    ----------
    osm_path
        Path to the OSM XML (or .osm.pbf) extract.
    output_net
        Destination ``.net.xml`` path. Parent directories are created.
    netconvert_binary
        Path or name of the ``netconvert`` executable. Defaults to
        ``$SUMO_HOME/bin/netconvert`` if ``SUMO_HOME`` is set, otherwise
        whatever is first on ``PATH``.
    road_types
        OSM road classes to keep (see :data:`DEFAULT_ROAD_TYPES`).
    extra_args
        Additional command-line arguments passed verbatim to
        ``netconvert``. Allows callers to override any default we set.
    runner
        Injected subprocess runner — defaults to :func:`subprocess.run`.
        Tests pass a fake to avoid touching the real binary.

    Returns
    -------
    A :class:`NetconvertResult` carrying the output path and the
    captured stdout/stderr. Raises :class:`NetconvertError` on failure.
    """
    osm_path = Path(osm_path)
    output_net = Path(output_net)
    if not osm_path.exists():
        raise FileNotFoundError(f"OSM input does not exist: {osm_path}")
    output_net.parent.mkdir(parents=True, exist_ok=True)

    binary = netconvert_binary or _resolve_netconvert_binary()

    command: list[str] = [
        binary,
        "--osm-files",
        str(osm_path),
        "--output-file",
        str(output_net),
        "--keep-edges.by-vclass",
        "passenger",
        "--type-files.keep",
        ",".join(road_types),
        # Geometry / topology cleanups that consistently improve quality.
        "--remove-edges.isolated",
        "true",
        "--geometry.remove",
        "true",
        "--ramps.guess",
        "true",
        # Traffic lights — import what OSM has and tidy the rest.
        "--tls.guess",
        "true",
        "--tls.discard-simple",
        "true",
        "--junctions.join",
        "true",
        # Disable progress noise.
        "--no-step-log",
        "true",
        *extra_args,
    ]

    completed = runner(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise NetconvertError(
            returncode=completed.returncode,
            stderr=completed.stderr or "",
            command=command,
        )

    return NetconvertResult(
        net_xml=output_net,
        command=tuple(command),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _resolve_netconvert_binary() -> str:
    """Return the best guess at the netconvert executable path.

    Order: SUMO_HOME/bin/netconvert (if SUMO_HOME is set), then a plain
    ``netconvert`` on PATH. Raises if neither is available.
    """
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / "netconvert"
        if candidate.exists():
            return str(candidate)
        # On Linux SUMO ships netconvert in /usr/share/sumo/bin (PPA)
        # but the binary lives in /usr/bin; check there too.
        for fallback in ("/usr/bin/netconvert",):
            if Path(fallback).exists():
                return fallback

    on_path = shutil.which("netconvert")
    if on_path:
        return on_path

    raise FileNotFoundError(
        "Could not locate the 'netconvert' binary. Install SUMO (>=1.20) "
        "and set SUMO_HOME, or ensure netconvert is on PATH."
    )


__all__ = [
    "DEFAULT_ROAD_TYPES",
    "NetconvertError",
    "NetconvertResult",
    "build_network_from_osm",
]
