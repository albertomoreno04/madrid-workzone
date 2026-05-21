"""Snap usable Madrid detectors to their nearest SUMO edges.

Bridges the data layer to the simulation layer: the audit script produced
``usable_detectors.json``; this script takes that list, looks up each
detector's coordinates from the latest snapshot, and matches every one
to its nearest edge in the SUMO ``.net.xml``. The output mapping is what
the OD-calibration pipeline reads to know which edge each observed flow
target applies to.

Outputs:
  data/processed/detectors/detector_edge_mapping.json   per-detector mapping
  data/processed/detectors/snap_report.json             aggregate stats
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from madrid_twin.config import DATA_PROCESSED, DATA_RAW
from madrid_twin.data.open_data import parse_traffic_intensity_xml
from madrid_twin.sim.snap import (
    SnapConfig,
    filter_to_ids,
    load_sumo_network,
    load_usable_detector_ids,
    pyproj_utm25830_to_wgs84,
    snap_detectors_to_edges,
    write_snap_outputs,
)


def _find_latest_snapshot(snapshot_dir: Path) -> Path | None:
    paths = sorted(glob.glob(str(snapshot_dir / "pm_*.xml")))
    return Path(paths[-1]) if paths else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--net",
        type=Path,
        default=DATA_PROCESSED / "network" / "madrid_m30_inner.net.xml",
        help="SUMO .net.xml path",
    )
    parser.add_argument(
        "--usable",
        type=Path,
        default=DATA_PROCESSED / "detectors" / "usable_detectors.json",
        help="Path to usable_detectors.json (from audit-detectors)",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Specific pm_*.xml snapshot to use (default: latest in data/raw/traffic_intensity)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_PROCESSED / "detectors",
        help="Output directory",
    )
    parser.add_argument(
        "--max-distance-m",
        type=float,
        default=100.0,
        help="Reject detector→edge matches farther than this (default 100 m)",
    )
    args = parser.parse_args()

    if not args.net.exists():
        print(f"SUMO network not found: {args.net}")
        print("Run `make build-network` first.")
        return

    if not args.usable.exists():
        print(f"Usable-detectors list not found: {args.usable}")
        print("Run `make audit-detectors` first (and collect enough probe snapshots).")
        return

    snapshot_path = args.snapshot or _find_latest_snapshot(DATA_RAW / "traffic_intensity")
    if snapshot_path is None or not snapshot_path.exists():
        print(f"No snapshot found in {DATA_RAW / 'traffic_intensity'}")
        print("Run `make probe` (or wait for the scheduled task) to collect at least one.")
        return

    print(f"Network:  {args.net}")
    print(f"Usable:   {args.usable}")
    print(f"Snapshot: {snapshot_path}")

    usable_ids = load_usable_detector_ids(args.usable)
    print(f"  usable detectors: {len(usable_ids)}")

    readings = parse_traffic_intensity_xml(snapshot_path.read_text(encoding="utf-8"))
    filtered = filter_to_ids(readings, usable_ids)
    print(f"  readings in snapshot intersecting usable set: {len(filtered)}")

    if not filtered:
        print("  no overlap between snapshot detectors and usable set — aborting")
        return

    print("Loading SUMO network (this can take a minute on a city-scale net)...")
    network = load_sumo_network(args.net)
    utm_to_lonlat = pyproj_utm25830_to_wgs84()

    print("Snapping detectors to nearest edges...")
    matches, report = snap_detectors_to_edges(
        filtered,
        network=network,
        utm_to_lonlat=utm_to_lonlat,
        config=SnapConfig(max_distance_m=args.max_distance_m),
    )

    print()
    print("--- snap report ---")
    print(json.dumps(report.to_dict(), indent=2))

    mapping_path, report_path = write_snap_outputs(matches, report, args.out_dir)
    print()
    print(f"Wrote: {mapping_path}")
    print(f"Wrote: {report_path}")

    # Print a small preview of the mapping.
    print()
    print("--- sample (first 5 matches) ---")
    for m in matches[:5]:
        print(
            f"  {m.detector_id} -> {m.edge_id}  "
            f"({m.distance_m:.1f} m"
            f"{', ' + m.description if m.description else ''})"
        )


if __name__ == "__main__":
    main()
