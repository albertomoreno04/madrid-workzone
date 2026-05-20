"""Madrid Ayuntamiento open-data clients.

The live ``informo.madrid.es/informo/tmadrid/pm.xml`` feed is richer than
a textbook traffic-detector dump: each ``<pm>`` element carries the
detector identifier (``<idelem>``), a human-readable description
(``<descripcion>``), the current intensity / occupancy / load
(``<intensidad>``, ``<ocupacion>``, ``<carga>``), the saturation /
capacity intensity (``<intensidadSat>``), a level-of-service code
(``<nivelServicio>``), an error flag (``<error>`` = ``N`` or ``Y``), an
administrative subarea, and **UTM coordinates** (``<st_x>``, ``<st_y>``
in EPSG:25830). A single ``<fecha_hora>`` element at the root of
``<pms>`` carries the snapshot timestamp and applies to every detector
in that snapshot.

:class:`DetectorReading` captures all of those fields. The capacity and
coordinate fields are essential for the Phase 4 W-SPSA calibration —
they let us geolocate detectors to network edges without a separate
location feed, and they give us per-detector capacities for the
count-matching loss denominators.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------

TRAFFIC_INTENSITY_URL: str = "https://informo.madrid.es/informo/tmadrid/pm.xml"

DEFAULT_USER_AGENT: str = (
    "madrid-workzone-digital-twin/0.1 "
    "(research; +https://github.com/albertomoreno04/madrid-workzone)"
)

DEFAULT_TIMEOUT_S: float = 15.0
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_BACKOFF_S: float = 1.5


# ---------------------------------------------------------------------------
# Data classes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorReading:
    """One detector's snapshot from the real-time intensity feed.

    The first six fields existed in the v0.1 parser; the remainder were
    added once we discovered the live feed is far richer than the
    initial test fixture suggested. All new fields default to ``None``
    so older / minimal XMLs still parse cleanly.
    """

    detector_id: str
    timestamp: datetime | None
    intensity_veh_h: float | None
    occupancy_pct: float | None
    load_pct: float | None
    # In the live feed this is derived from <error>: 'N' -> '0' (healthy),
    # 'Y' -> '1' (failure). In legacy fixtures it can come directly from
    # <st_intensidad>.
    service_status: str | None

    description: str | None = None
    # Saturation intensity / per-detector capacity, veh/h. Used directly
    # as the denominator in the W-SPSA count-matching loss.
    intensity_sat_veh_h: float | None = None
    # Level-of-service code reported by the operator (e.g. "0" through "3").
    service_level: str | None = None
    # Administrative subarea string used by the Ayuntamiento for grouping.
    subarea: str | None = None
    # UTM coordinates (EPSG:25830, the official CRS for peninsular Spain).
    # Lets us snap detectors to SUMO edges without a separate location feed.
    x_utm: float | None = None
    y_utm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return d


# ---------------------------------------------------------------------------
# Parsing helpers.
# ---------------------------------------------------------------------------


def _parse_madrid_timestamp(raw: str | None) -> datetime | None:
    """Parse the Ayuntamiento timestamp; tolerant of ISO 8601."""
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%d/%m/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _maybe_float(raw: str | None) -> float | None:
    """Parse a numeric field, tolerating Spanish decimal commas."""
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return None


def _maybe_str(raw: str | None) -> str | None:
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None


def _error_flag_to_status(error_flag: str | None) -> str | None:
    """Map the Ayuntamiento ``<error>`` value to our service-status code.

    ``N`` (no error) becomes ``"0"`` (healthy), ``Y`` (error) becomes
    ``"1"`` (failure). Any other value — empty, missing, garbled — yields
    ``None`` so callers fall back to legacy fields.
    """
    if not error_flag:
        return None
    f = error_flag.strip().upper()
    if f == "N":
        return "0"
    if f == "Y":
        return "1"
    return None


# ---------------------------------------------------------------------------
# Parsing.
# ---------------------------------------------------------------------------


def parse_traffic_intensity_xml(xml_text: str) -> list[DetectorReading]:
    """Parse the Ayuntamiento real-time XML into typed detector readings.

    Handles both XML shapes seen in the wild:

    - The **live feed** at ``informo.madrid.es/informo/tmadrid/pm.xml``:
      one ``<fecha_hora>`` at the root that applies to all ``<pm>``
      children, ``<error>`` flag per detector, plus ``<descripcion>``,
      ``<intensidadSat>``, ``<nivelServicio>``, ``<subarea>``, ``<st_x>``
      and ``<st_y>``.
    - **Legacy / fixture** XMLs with per-``<pm>`` ``<fecha_hora>`` and
      ``<st_intensidad>`` as the health flag.

    A per-``<pm>`` ``<fecha_hora>`` takes precedence over the root one
    when both are present. ``<error>`` takes precedence over
    ``<st_intensidad>`` for the same reason.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed traffic-intensity XML: {exc}") from exc

    # Snapshot-wide timestamp from the root, if present.
    root_fecha_hora_el = root.find("fecha_hora")
    root_timestamp = _parse_madrid_timestamp(
        root_fecha_hora_el.text if root_fecha_hora_el is not None else None
    )

    readings: list[DetectorReading] = []
    candidates = list(root.iter("pm"))
    if not candidates:
        candidates = list(root.iter("punto-medida"))

    for pm in candidates:
        detector_id = (pm.findtext("idelem") or pm.findtext("id") or "").strip()
        if not detector_id:
            continue

        # Timestamp resolution: prefer per-pm, fall back to root.
        per_pm_timestamp = _parse_madrid_timestamp(pm.findtext("fecha_hora"))
        timestamp = per_pm_timestamp or root_timestamp

        # Health flag: prefer <error>, fall back to legacy <st_intensidad>.
        error_flag = _maybe_str(pm.findtext("error"))
        status_from_error = _error_flag_to_status(error_flag)
        legacy_status = _maybe_str(pm.findtext("st_intensidad"))
        service_status = status_from_error if status_from_error is not None else legacy_status

        readings.append(
            DetectorReading(
                detector_id=detector_id,
                timestamp=timestamp,
                intensity_veh_h=_maybe_float(pm.findtext("intensidad")),
                occupancy_pct=_maybe_float(pm.findtext("ocupacion")),
                load_pct=_maybe_float(pm.findtext("carga")),
                service_status=service_status,
                description=_maybe_str(pm.findtext("descripcion")),
                intensity_sat_veh_h=_maybe_float(pm.findtext("intensidadSat")),
                service_level=_maybe_str(pm.findtext("nivelServicio")),
                subarea=_maybe_str(pm.findtext("subarea")),
                x_utm=_maybe_float(pm.findtext("st_x")),
                y_utm=_maybe_float(pm.findtext("st_y")),
            )
        )

    return readings


# ---------------------------------------------------------------------------
# HTTP fetch with retry / cache.
# ---------------------------------------------------------------------------


def fetch_traffic_intensity_xml(
    url: str = TRAFFIC_INTENSITY_URL,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
    user_agent: str = DEFAULT_USER_AGENT,
    cache_path: Path | None = None,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Fetch the Madrid real-time traffic intensity XML payload with retries."""
    last_error: Exception | None = None
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/xml, text/xml, */*;q=0.1",
    }
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout_s, headers=headers)

    try:
        for attempt in range(max_retries):
            try:
                response = http.get(url, headers=headers)
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"server returned {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                xml_text = response.text
                if cache_path is not None:
                    cache_path = Path(cache_path)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(xml_text, encoding="utf-8")
                return xml_text
            except (httpx.HTTPError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt + 1 < max_retries:
                    sleep(backoff_s ** (attempt + 1))
                continue

        raise RuntimeError(
            f"fetch_traffic_intensity_xml: exhausted {max_retries} attempts against {url}"
        ) from last_error
    finally:
        if owns_client:
            http.close()


def fetch_traffic_intensity(
    url: str = TRAFFIC_INTENSITY_URL,
    **kwargs: Any,
) -> list[DetectorReading]:
    """Convenience: fetch + parse in one call."""
    xml_text = fetch_traffic_intensity_xml(url, **kwargs)
    return parse_traffic_intensity_xml(xml_text)


# ---------------------------------------------------------------------------
# Aggregation helpers.
# ---------------------------------------------------------------------------


def filter_valid_readings(readings: Iterable[DetectorReading]) -> list[DetectorReading]:
    """Return only readings whose service status indicates a healthy detector."""
    out: list[DetectorReading] = []
    for r in readings:
        if r.service_status not in (None, "", "0"):
            continue
        if r.intensity_veh_h is None:
            continue
        out.append(r)
    return out


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_BACKOFF_S",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_USER_AGENT",
    "DetectorReading",
    "TRAFFIC_INTENSITY_URL",
    "fetch_traffic_intensity",
    "fetch_traffic_intensity_xml",
    "filter_valid_readings",
    "parse_traffic_intensity_xml",
]
