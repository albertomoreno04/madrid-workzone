"""Decompose the SUMO network into Traffic Analysis Zones (TAZ).

K-means++ on edge centroids. ``n_zones=60`` by default — small enough
that the resulting OD matrix (60² = 3600 cells) is tractable for W-SPSA,
big enough that each zone covers ~1.5 km² in inner Madrid.

Outputs:
  data/processed/taz/taz.xml                SUMO TAZ XML (duarouter input)
  data/processed/taz/edge_to_taz.json       per-edge zone mapping (debug/audit)
  data/processed/taz/taz_report.json        aggregate stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from madrid_twin.config import DATA_PROCESSED
from madrid_twin.sim.taz import (
    build_taz_by_kmeans,
    extract_edge_centroids_from_sumo_network,
    write_taz_xml,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--net",
        type=Path,
        default=DATA_PROCESSED / "network" / "madrid_m30_inner.net.xml",
        help="SUMO .net.xml path",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_PROCESSED / "taz",
        help="Output directory",
    )
    parser.add_argument(
        "--n-zones",
        type=int,
        default=60,
        help="Target number of TAZ zones (default 60)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="K-means++ random seed for reproducibility",
    )
    args = parser.parse_args()

    if not args.net.exists():
        print(f"SUMO network not found: {args.net}")
        print("Run `make build-network` first.")
        return

    print(f"Network: {args.net}")
    print("Extracting edge centroids (this can take ~30s on a city-scale net)...")
    edges = extract_edge_centroids_from_sumo_network(args.net)
    print(f"  edges with valid geometry: {len(edges)}")

    print(f"Clustering into {args.n_zones} zones (K-means++, seed={args.seed})...")
    mapping, report = build_taz_by_kmeans(edges, n_zones=args.n_zones, seed=args.seed)

    print()
    print("--- TAZ report ---")
    print(json.dumps(report.to_dict(), indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    xml_path = args.out_dir / "taz.xml"
    write_taz_xml(mapping, xml_path)

    json_path = args.out_dir / "edge_to_taz.json"
    json_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    report_path = args.out_dir / "taz_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    print()
    print(f"Wrote: {xml_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
