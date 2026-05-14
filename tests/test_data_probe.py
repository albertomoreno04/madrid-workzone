"""Tests for the open-data probe."""

from __future__ import annotations

from pathlib import Path

import httpx

from madrid_twin.data.probe import ProbeReport, run_probe


def test_run_probe_returns_structured_report(
    tmp_path: Path, traffic_intensity_sample_xml: str
) -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, text=traffic_intensity_sample_xml)
    )
    client = httpx.Client(transport=transport, timeout=5.0)

    snapshot_dir = tmp_path / "snapshots"
    report = run_probe(
        snapshot_dir=snapshot_dir,
        client=client,
    )

    assert isinstance(report, ProbeReport)
    assert report.n_detectors == 3
    assert report.n_with_intensity == 2
    assert report.payload_bytes > 0
    assert report.elapsed_ms >= 0.0
    assert len(report.sample) == 3
    assert report.snapshot_path is not None
    assert Path(report.snapshot_path).exists()


def test_run_probe_without_snapshot_dir_skips_write(
    traffic_intensity_sample_xml: str,
) -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, text=traffic_intensity_sample_xml)
    )
    client = httpx.Client(transport=transport, timeout=5.0)

    report = run_probe(snapshot_dir=None, client=client)
    assert report.snapshot_path is None


def test_probe_report_kv_lines_format() -> None:
    report = ProbeReport(
        url="http://example.com",
        fetched_at_utc="2026-05-13T08:30:00+00:00",
        elapsed_ms=123.4,
        payload_bytes=4096,
        n_detectors=10,
        n_with_intensity=9,
        sample=[],
        snapshot_path=None,
    )
    s = report.to_kv_lines()
    assert "url=http://example.com" in s
    assert "elapsed_ms=123.4" in s
    assert "n_detectors=10" in s
