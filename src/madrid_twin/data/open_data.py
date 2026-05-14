"""Madrid Ayuntamiento open-data clients."""

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
    """One detector's snapshot from the real-time intensity feed."""

    detector_id: str
    timestamp: datetime | None
    intensity_veh_h: float | None
    occupancy_pct: float | None
    load_pct: float | None
    service_status: str | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return d


# ---------------------------------------------------------------------------
# Parsing.
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
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return None


def parse_traffic_intensity_xml(xml_text: str) -> list[DetectorReading]:
    """Parse the Ayuntamiento real-time XML into typed detector readings."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed traffic-intensity XML: {exc}") from exc

    readings: list[DetectorReading] = []
    candidates = list(root.iter("pm"))
    if not candidates:
        candidates = list(root.iter("punto-medida"))

    for pm in candidates:
        detector_id = (pm.findtext("idelem") or pm.findtext("id") or "").strip()
        if not detector_id:
            continue
        readings.append(
            DetectorReading(
                detector_id=detector_id,
                timestamp=_parse_madrid_timestamp(pm.findtext("fecha_hora")),
                intensity_veh_h=_maybe_float(pm.findtext("intensidad")),
                occupancy_pct=_maybe_float(pm.findtext("ocupacion")),
                load_pct=_maybe_float(pm.findtext("carga")),
                service_status=(pm.findtext("st_intensidad") or "").strip() or None,
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
