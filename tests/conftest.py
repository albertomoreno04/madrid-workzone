"""Shared pytest fixtures for MADTwin."""

from __future__ import annotations

from pathlib import Path

import pytest

SUMMARY_XML = """\
<summary>
  <step time="0.00" inserted="0" ended="0" running="0" halting="0"
        meanSpeed="0.0" meanWaitingTime="0.0" meanTravelTime="0.0"/>
  <step time="1.00" inserted="2" ended="0" running="2" halting="0"
        meanSpeed="10.0" meanWaitingTime="0.0" meanTravelTime="1.0"/>
  <step time="2.00" inserted="1" ended="1" running="2" halting="1"
        meanSpeed="8.0" meanWaitingTime="0.5" meanTravelTime="1.2"/>
  <step time="3.00" inserted="0" ended="2" running="0" halting="0"
        meanSpeed="0.0" meanWaitingTime="0.0" meanTravelTime="0.0"/>
</summary>
"""

TRIPINFO_XML = """\
<tripinfos>
  <tripinfo id="v0" duration="100" routeLength="1500" waitingTime="10"
            timeLoss="20" departDelay="0" waitingCount="2" stopTime="5"/>
  <tripinfo id="v1" duration="200" routeLength="2500" waitingTime="40"
            timeLoss="60" departDelay="5" waitingCount="4" stopTime="15"/>
  <tripinfo id="v2" duration="300" routeLength="3500" waitingTime="80"
            timeLoss="120" departDelay="10" waitingCount="6" stopTime="25"/>
</tripinfos>
"""

STATISTICS_XML = """\
<statistics>
  <vehicles loaded="3" inserted="3" running="0" waiting="0"/>
  <teleports total="0" jam="0" yield="0" wrongLane="0"/>
  <safety collisions="0" emergencyStops="0"/>
</statistics>
"""

# Minimal / legacy-shaped fixture: per-<pm> timestamps, <st_intensidad>
# as the health flag. Kept verbatim because earlier tests assert against
# its specific values.
TRAFFIC_INTENSITY_SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<pms>
  <pm>
    <idelem>3501</idelem>
    <fecha_hora>13/05/2026 08:30:00</fecha_hora>
    <intensidad>1240</intensidad>
    <ocupacion>12,5</ocupacion>
    <carga>34</carga>
    <st_intensidad>0</st_intensidad>
  </pm>
  <pm>
    <idelem>3502</idelem>
    <fecha_hora>13/05/2026 08:30:00</fecha_hora>
    <intensidad>780</intensidad>
    <ocupacion>7,2</ocupacion>
    <carga>18</carga>
    <st_intensidad>0</st_intensidad>
  </pm>
  <pm>
    <idelem>3503</idelem>
    <fecha_hora>13/05/2026 08:30:00</fecha_hora>
    <intensidad></intensidad>
    <ocupacion></ocupacion>
    <carga></carga>
    <st_intensidad>2</st_intensidad>
  </pm>
</pms>
"""

# Real-feed-shaped fixture: snapshot-wide <fecha_hora> at the root,
# <error>N|Y</error> as the health flag, and the rich detector
# metadata (descripcion, intensidadSat, nivelServicio, subarea, st_x, st_y).
# Mirrors the actual informo.madrid.es/informo/tmadrid/pm.xml structure.
REAL_TRAFFIC_INTENSITY_SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<pms>
  <fecha_hora>20/05/2026 12:50:19</fecha_hora>
  <pm>
    <idelem>9841</idelem>
    <descripcion>Valle de Mena S-E - Acc.Ramon Castroviejo</descripcion>
    <accesoAsociado>0301005</accesoAsociado>
    <intensidad>400</intensidad>
    <ocupacion>3</ocupacion>
    <carga>16</carga>
    <nivelServicio>0</nivelServicio>
    <intensidadSat>3100</intensidadSat>
    <error>N</error>
    <subarea>0328</subarea>
    <st_x>438339,375874991</st_x>
    <st_y>4480454,96970565</st_y>
  </pm>
  <pm>
    <idelem>9842</idelem>
    <descripcion>Valle de Mena N-O - Acc.Ramon Castroviejo</descripcion>
    <accesoAsociado>0301006</accesoAsociado>
    <intensidad>240</intensidad>
    <ocupacion>2</ocupacion>
    <carga>11</carga>
    <nivelServicio>0</nivelServicio>
    <intensidadSat>3100</intensidadSat>
    <error>N</error>
    <subarea>0328</subarea>
    <st_x>438345,5</st_x>
    <st_y>4480460,8</st_y>
  </pm>
  <pm>
    <idelem>9999</idelem>
    <descripcion>Broken detector</descripcion>
    <accesoAsociado>0999999</accesoAsociado>
    <intensidad></intensidad>
    <ocupacion></ocupacion>
    <carga></carga>
    <nivelServicio>0</nivelServicio>
    <intensidadSat>3100</intensidadSat>
    <error>Y</error>
    <subarea>0999</subarea>
    <st_x>0</st_x>
    <st_y>0</st_y>
  </pm>
</pms>
"""


@pytest.fixture()
def traffic_intensity_sample_xml() -> str:
    """Shared XML payload mirroring the legacy fixture format."""
    return TRAFFIC_INTENSITY_SAMPLE_XML


@pytest.fixture()
def real_traffic_intensity_sample_xml() -> str:
    """Shared XML payload mirroring the live Ayuntamiento feed format."""
    return REAL_TRAFFIC_INTENSITY_SAMPLE_XML


@pytest.fixture()
def sumo_outputs(tmp_path: Path) -> dict[str, Path]:
    """Write a small synthetic SUMO output bundle to disk and return paths."""
    summary = tmp_path / "summary.xml"
    tripinfo = tmp_path / "tripinfo.xml"
    stats = tmp_path / "statistics.xml"

    summary.write_text(SUMMARY_XML, encoding="utf-8")
    tripinfo.write_text(TRIPINFO_XML, encoding="utf-8")
    stats.write_text(STATISTICS_XML, encoding="utf-8")

    return {"summary": summary, "tripinfo": tripinfo, "statistics": stats}
