"""Build a SUMO network from an OSM extract — Week 1 of the research workflow.

Wraps :func:`madrid_twin.sim.network.build_network_from_osm` with the
Madrid-specific defaults documented in the research-workflow guide. The
default input is ``data/external/osm/madrid_m30_inner.osm`` (which you
must download yourself — see the Overpass query in the README). The
default output is ``data/processed/network/madrid_m30_inner.net.xml``.

Override either with the ``--osm`` and ``--out`` CLI flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from madrid_twin.config import DATA_EXTERNAL, DATA_PROCESSED
from madrid_twin.sim.network import build_network_from_osm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--osm",
        type=Path,
        default=DATA_EXTERNAL / "osm" / "madrid_m30_inner.osm",
        help="Input OSM extract (default: data/external/osm/madrid_m30_inner.osm)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_PROCESSED / "network" / "madrid_m30_inner.net.xml",
        help="Output SUMO net.xml path",
    )
    parser.add_argument(
        "--netconvert-binary",
        type=str,
        default=None,
        help="Override the path to the netconvert binary",
    )
    args = parser.parse_args()

    if not args.osm.exists():
        print(f"OSM extract not found: {args.osm}")
        print()
        print("Download an extract first. Suggested workflow:")
        print("  1. Visit https://overpass-turbo.eu/")
        print("  2. Paste the Overpass query from the research-workflow guide")
        print(f"  3. Export -> Save as -> raw OSM data, save to {args.osm}")
        return

    print(f"Building SUMO network from: {args.osm}")
    print(f"Output: {args.out}")
    result = build_network_from_osm(
        args.osm,
        args.out,
        netconvert_binary=args.netconvert_binary,
    )
    print()
    print("netconvert finished successfully.")
    print(f"  command: {' '.join(result.command)}")
    if result.stdout:
        tail = "\n".join(result.stdout.strip().splitlines()[-10:])
        print(f"  stdout (last 10 lines):\n{tail}")


if __name__ == "__main__":
    main()
