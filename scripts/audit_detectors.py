"""Week 1 audit: per-detector reliability over accumulated probe snapshots.

Run this once you've been scheduling ``make probe`` every 5 minutes for
at least a few days. The longer the window, the more reliable the
estimates — two weeks of snapshots is the recommended minimum before
making "this detector is usable" decisions.

Outputs:
  data/processed/detectors/detector_reliability.json  full per-detector table
  data/processed/detectors/usable_detectors.json      filtered ID list
"""

from __future__ import annotations

import json

from madrid_twin.config import DATA_PROCESSED, DATA_RAW
from madrid_twin.data.audit import (
    AuditConfig,
    aggregate_snapshots,
    filter_usable_detectors,
    load_snapshot_files,
    write_audit_outputs,
)


def main() -> None:
    snapshot_dir = DATA_RAW / "traffic_intensity"
    output_dir = DATA_PROCESSED / "detectors"

    print(f"Scanning snapshots in: {snapshot_dir}")
    snapshots = load_snapshot_files(snapshot_dir)
    print(f"  loaded {len(snapshots)} snapshot files")

    if not snapshots:
        print("  no snapshots found — schedule `make probe` and wait a few days")
        return

    reliabilities = aggregate_snapshots(snapshots)
    print(f"  unique detectors seen: {len(reliabilities)}")

    cfg = AuditConfig()
    usable = filter_usable_detectors(reliabilities.values(), cfg)
    print(
        f"  detectors meeting reliability bar "
        f"(>= {cfg.min_healthy_fraction:.0%} healthy, "
        f">= {cfg.min_intensity_fraction:.0%} with intensity, "
        f"mean intensity >= {cfg.min_mean_intensity_veh_h:g} veh/h): "
        f"{len(usable)}"
    )

    table_path, usable_path = write_audit_outputs(reliabilities, usable, output_dir)
    print()
    print(f"Wrote: {table_path}")
    print(f"Wrote: {usable_path}")

    # Print a small summary preview.
    sample = list(reliabilities.values())[:5]
    print()
    print("--- sample (first 5) ---")
    for r in sample:
        print(
            json.dumps(
                {
                    "detector_id": r.detector_id,
                    "healthy_fraction": round(r.healthy_fraction, 3),
                    "intensity_fraction": round(r.intensity_fraction, 3),
                    "mean_intensity_veh_h": (
                        round(r.mean_intensity_veh_h, 1)
                        if r.mean_intensity_veh_h is not None
                        else None
                    ),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
