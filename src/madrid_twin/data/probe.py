"""Open-data smoke probe.

A small, one-shot diagnostic that confirms we can pull the Madrid
real-time traffic intensity feed at the cadence we need.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from madrid_twin.config import DATA_RAW
from madrid_twin.data.open_data import (
    TRAFFIC_INTENSITY_URL,
    fetch_traffic_intensity_xml,
    parse_traffic_intensity_xml,
)

DEFAULT_SAMPLE_SIZE: int = 3


@dataclass(frozen=True)
class ProbeReport:
    """Outcome of a single probe run, ready for JSON serialization."""

    url: str
    fetched_at_utc: str
    elapsed_ms: float
    payload_bytes: int
    n_detectors: int
    n_with_intensity: int
    sample: list[dict[str, Any]]
    snapshot_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_kv_lines(self) -> str:
        return "\n".join(
            [
                f"url={self.url}",
                f"fetched_at_utc={self.fetched_at_utc}",
                f"elapsed_ms={self.elapsed_ms:.1f}",
                f"payload_bytes={self.payload_bytes}",
                f"n_detectors={self.n_detectors}",
                f"n_with_intensity={self.n_with_intensity}",
                f"snapshot_path={self.snapshot_path}",
            ]
        )


def run_probe(
    url: str = TRAFFIC_INTENSITY_URL,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    snapshot_dir: Path | None = None,
    **fetch_kwargs: Any,
) -> ProbeReport:
    """Perform one probe of the traffic intensity feed."""
    # UP017: keep timezone.utc rather than the 3.11-only datetime.UTC alias
    # so the module imports cleanly on Python 3.10 too (used in CI sandbox).
    started_utc = datetime.now(timezone.utc)  # noqa: UP017
    t0 = time.perf_counter()

    snapshot_path: Path | None = None
    if snapshot_dir is not None:
        snapshot_dir = Path(snapshot_dir)
        snapshot_path = snapshot_dir / f"pm_{started_utc.strftime('%Y%m%dT%H%M%SZ')}.xml"

    xml_text = fetch_traffic_intensity_xml(url, cache_path=snapshot_path, **fetch_kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    readings = parse_traffic_intensity_xml(xml_text)
    n_with_intensity = sum(1 for r in readings if r.intensity_veh_h is not None)
    sample = [r.to_dict() for r in readings[:sample_size]]

    return ProbeReport(
        url=url,
        fetched_at_utc=started_utc.isoformat(),
        elapsed_ms=elapsed_ms,
        payload_bytes=len(xml_text.encode("utf-8")),
        n_detectors=len(readings),
        n_with_intensity=n_with_intensity,
        sample=sample,
        snapshot_path=str(snapshot_path) if snapshot_path else None,
    )


def main() -> None:
    """CLI entry point used by ``scripts/probe_open_data.py``."""
    snapshot_dir = DATA_RAW / "traffic_intensity"
    report = run_probe(snapshot_dir=snapshot_dir)
    print(report.to_kv_lines())
    print()
    print("--- sample (first 3) ---")
    for entry in report.sample:
        print(json.dumps(entry, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_SAMPLE_SIZE",
    "ProbeReport",
    "main",
    "run_probe",
]
