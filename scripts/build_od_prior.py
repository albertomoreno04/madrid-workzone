"""Build the gravity-model OD prior matrix.

Combines the TAZ decomposition, the snapped detector→edge mapping, and
the latest observed snapshot to produce a 60×60 (or however many
zones × zones) seed matrix for W-SPSA calibration. Zone "size" is
proxied by observed detector flow in the zone; deterrence is
exponential in inter-centroid distance.

Outputs:
  data/processed/od/od_prior.npy        the (K, K) matrix
  data/processed/od/zones.json          ordered zone metadata (row index = zone)
  data/processed/od/od_prior_report.json summary stats
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
)
from madrid_twin.sim.od_prior import (
    aggregate_zone_stats,
    build_od_prior_report,
    gravity_od_prior,
    save_od_prior,
)
from madrid_twin.sim.taz import extract_edge_centroids_from_sumo_network


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
        "--taz",
        type=Path,
        default=DATA_PROCESSED / "taz" / "edge_to_taz.json",
        help="Edge-to-TAZ mapping JSON (from build_taz)",
    )
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
        help="pm_*.xml snapshot (default: latest in data/raw/traffic_intensity)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_PROCESSED / "od",
        help="Output directory",
    )
    parser.add_argument(
        "--beta-per-km",
        type=float,
        default=0.1,
        help="Gravity deterrence parameter, 1/km (default 0.1 — typical urban)",
    )
    args = parser.parse_args()

    for path, label in [
        (args.net, "SUMO network"),
        (args.taz, "TAZ mapping"),
        (args.mapping, "detector-edge mapping"),
    ]:
        if not path.exists():
            print(f"{label} not found: {path}")
            return

    snapshot_path = args.snapshot or _find_latest_snapshot(DATA_RAW / "traffic_intensity")
    if snapshot_path is None or not snapshot_path.exists():
        print(f"No snapshot found in {DATA_RAW / 'traffic_intensity'}")
        return

    print(f"Network:  {args.net}")
    print(f"TAZ:      {args.taz}")
    print(f"Mapping:  {args.mapping}")
    print(f"Snapshot: {snapshot_path}")

    print("Extracting edge centroids...")
    edges = extract_edge_centroids_from_sumo_network(args.net)
    edge_centroids = {edge_id: (cx, cy) for edge_id, cx, cy in edges}
    print(f"  edges: {len(edge_centroids)}")

    edge_to_zone = {k: int(v) for k, v in json.loads(args.taz.read_text(encoding="utf-8")).items()}
    print(f"  edges with zone assignment: {len(edge_to_zone)}")

    detector_to_edge = load_detector_edge_mapping(args.mapping)
    print(f"  snapped detectors: {len(detector_to_edge)}")

    readings = parse_traffic_intensity_xml(snapshot_path.read_text(encoding="utf-8"))
    observations = aggregate_observations_by_edge(readings, detector_to_edge)
    print(f"  edges with observed flow: {len(observations)}")

    print("Aggregating to zones...")
    zone_stats = aggregate_zone_stats(edge_to_zone, edge_centroids, observations)
    zones_ordered = [zone_stats[z] for z in sorted(zone_stats)]
    print(f"  zones with stats: {len(zones_ordered)}")
    print(
        f"  zones with non-zero observed flow: "
        f"{sum(1 for z in zones_ordered if z.total_observed_flow_veh_h > 0)}"
    )

    print(f"Building gravity OD prior (beta = {args.beta_per_km}/km)...")
    matrix = gravity_od_prior(zones_ordered, beta_per_km=args.beta_per_km)

    report = build_od_prior_report(matrix, args.beta_per_km)
    print()
    print("--- OD prior report ---")
    print(json.dumps(report.to_dict(), indent=2))

    matrix_path, zones_path, report_path = save_od_prior(
        matrix, zones_ordered, args.out_dir, beta_per_km=args.beta_per_km
    )
    print()
    print(f"Wrote: {matrix_path}")
    print(f"Wrote: {zones_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
