"""Emit a SUMO edgeData XML from the latest detector snapshot.

Joins the latest ``pm_*.xml`` snapshot to ``detector_edge_mapping.json``
(produced by :mod:`scripts.snap_detectors`) and writes a SUMO meandata
file that ``sumo-gui`` can color the network by. This is the first time
real Madrid flows show up on the simulation network — useful as a
sanity-check before the W-SPSA calibration loop.

Output: ``data/processed/network/observed_flows.edgedata.xml``

Visualizing in sumo-gui:
  1. ``sumo-gui -n data/processed/network/madrid_m30_inner.net.xml``
  2. Menu: File → Open Edge Data → select ``observed_flows.edgedata.xml``
  3. Menu: View → Edit Visualization → Streets → Color by →
     "edgeData attribute" → pick ``intensity_veh_h``.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from madrid_twin.config import DATA_PROCESSED, DATA_RAW
from madrid_twin.data.open_data import parse_traffic_intensity_xml
from madrid_twin.sim.edgedata import (
    aggregate_observations_by_edge,
    load_detector_edge_mapping,
    write_edgedata_xml,
)


def _find_latest_snapshot(snapshot_dir: Path) -> Path | None:
    paths = sorted(glob.glob(str(snapshot_dir / "pm_*.xml")))
    return Path(paths[-1]) if paths else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DATA_PROCESSED / "detectors" / "detector_edge_mapping.json",
        help="Detector-to-edge mapping JSON (from snap_detectors)",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="pm_*.xml snapshot to use (default: latest in data/raw/traffic_intensity)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_PROCESSED / "network" / "observed_flows.edgedata.xml",
        help="Output edgeData XML path",
    )
    parser.add_argument(
        "--interval-id",
        type=str,
        default="observed",
        help="ID for the <interval> element (default: 'observed')",
    )
    args = parser.parse_args()

    if not args.mapping.exists():
        print(f"Detector-edge mapping not found: {args.mapping}")
        print("Run `make snap-detectors` first.")
        return

    snapshot_path = args.snapshot or _find_latest_snapshot(DATA_RAW / "traffic_intensity")
    if snapshot_path is None or not snapshot_path.exists():
        print(f"No snapshot found in {DATA_RAW / 'traffic_intensity'}")
        return

    print(f"Mapping:  {args.mapping}")
    print(f"Snapshot: {snapshot_path}")

    mapping = load_detector_edge_mapping(args.mapping)
    print(f"  detectors with edge mapping: {len(mapping)}")

    readings = parse_traffic_intensity_xml(snapshot_path.read_text(encoding="utf-8"))
    print(f"  detector readings in snapshot: {len(readings)}")

    observations = aggregate_observations_by_edge(readings, mapping)
    print(f"  edges with observed flow: {len(observations)}")

    if not observations:
        print("  no overlap between snapshot and mapping — aborting")
        return

    out_path = write_edgedata_xml(observations, args.out, interval_id=args.interval_id)
    print()
    print(f"Wrote: {out_path}")

    # Quick distribution summary.
    intensities = sorted(o.intensity_veh_h for o in observations.values())
    n = len(intensities)
    print()
    print("--- intensity distribution (veh/h) ---")
    print(
        json.dumps(
            {
                "min": round(intensities[0], 1),
                "p25": round(intensities[n // 4], 1),
                "median": round(intensities[n // 2], 1),
                "p75": round(intensities[3 * n // 4], 1),
                "max": round(intensities[-1], 1),
            },
            indent=2,
        )
    )

    # Top-10 busiest edges — should be M-30 / Castellana / Alcalá.
    top = sorted(observations.values(), key=lambda o: o.intensity_veh_h, reverse=True)[:10]
    print()
    print("--- top 10 busiest edges ---")
    for o in top:
        print(f"  {o.edge_id:>20}  {o.intensity_veh_h:>7.0f} veh/h  ({o.n_detectors} detector(s))")

    print()
    print("To visualize:")
    print("  sumo-gui -n data/processed/network/madrid_m30_inner.net.xml")
    print(f"  File menu -> Open Edge Data -> {out_path}")
    print("  View menu -> Edit Visualization -> Streets -> Color by -> edgeData: intensity_veh_h")


if __name__ == "__main__":
    main()
