"""Tests for the netconvert wrapper.

Subprocess is fully mocked — we never invoke the real ``netconvert``
binary. The wrapper's contract is: assemble a defensible command line,
run it, surface stderr on failure, return the output path on success.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from madrid_twin.sim.network import (
    DEFAULT_ROAD_TYPES,
    NetconvertError,
    NetconvertResult,
    build_network_from_osm,
)


def _make_runner(returncode: int = 0, stdout: str = "", stderr: str = ""):
    """Build a fake subprocess.run that records the call and returns the canned result."""
    recorded: list[list[str]] = []

    def runner(command, **kwargs):
        recorded.append(list(command))
        return subprocess.CompletedProcess(
            args=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    runner.recorded = recorded  # type: ignore[attr-defined]
    return runner


# ---------------------------------------------------------------------------
# Happy path.
# ---------------------------------------------------------------------------


class TestBuildNetwork:
    def test_happy_path_returns_result(self, tmp_path: Path) -> None:
        osm = tmp_path / "madrid.osm"
        osm.write_text("<osm/>", encoding="utf-8")
        out = tmp_path / "out" / "madrid.net.xml"

        runner = _make_runner(returncode=0, stdout="ok")
        result = build_network_from_osm(
            osm,
            out,
            netconvert_binary="netconvert",  # bypass auto-resolve
            runner=runner,
        )
        assert isinstance(result, NetconvertResult)
        assert result.net_xml == out
        assert "ok" in result.stdout
        assert out.parent.exists()  # parent dir was created

    def test_command_contains_expected_flags(self, tmp_path: Path) -> None:
        osm = tmp_path / "in.osm"
        osm.write_text("<osm/>", encoding="utf-8")
        out = tmp_path / "out.net.xml"
        runner = _make_runner()

        build_network_from_osm(osm, out, netconvert_binary="netconvert", runner=runner)

        assert len(runner.recorded) == 1
        cmd = runner.recorded[0]
        assert cmd[0] == "netconvert"
        assert "--osm-files" in cmd
        assert str(osm) in cmd
        assert "--output-file" in cmd
        assert str(out) in cmd
        # All default road types should appear comma-joined.
        types_index = cmd.index("--keep-edges.by-type") + 1
        assert "highway.motorway" in cmd[types_index]
        for road_type in DEFAULT_ROAD_TYPES:
            assert road_type in cmd[types_index]

    def test_extra_args_appended(self, tmp_path: Path) -> None:
        osm = tmp_path / "in.osm"
        osm.write_text("<osm/>", encoding="utf-8")
        out = tmp_path / "out.net.xml"
        runner = _make_runner()

        build_network_from_osm(
            osm,
            out,
            netconvert_binary="netconvert",
            extra_args=("--my-custom-flag", "value"),
            runner=runner,
        )
        cmd = runner.recorded[0]
        assert cmd[-2:] == ["--my-custom-flag", "value"]


# ---------------------------------------------------------------------------
# Error paths.
# ---------------------------------------------------------------------------


class TestBuildNetworkErrors:
    def test_missing_osm_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_network_from_osm(
                tmp_path / "missing.osm",
                tmp_path / "out.net.xml",
                netconvert_binary="netconvert",
                runner=_make_runner(),
            )

    def test_non_zero_exit_raises_netconvert_error(self, tmp_path: Path) -> None:
        osm = tmp_path / "in.osm"
        osm.write_text("<osm/>", encoding="utf-8")
        runner = _make_runner(returncode=2, stderr="bad input\n")

        with pytest.raises(NetconvertError) as excinfo:
            build_network_from_osm(
                osm,
                tmp_path / "out.net.xml",
                netconvert_binary="netconvert",
                runner=runner,
            )
        assert excinfo.value.returncode == 2
        assert "bad input" in excinfo.value.stderr
