"""
Geographic utilities: Haversine distance and bounding-box helpers.

Two filtering strategies are implemented to match different geographic scopes:

  city:       radius-based Haversine filter (fast, precise for local search)
  province:   bounding-box filter with an auto-calculated covering radius
              (necessary because a 50 km radius around Québec City barely
               covers a city, but a user searching "in Québec" expects the
               entire province)

The bounding boxes below are conservative estimates for Canadian provinces
most relevant to manufacturing search. Extend as needed.
"""
import math
from typing import Optional

# (lat_min, lat_max, lon_min, lon_max)
PROVINCE_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "QC": (44.99, 62.59, -79.76, -57.10),
    "ON": (41.67, 56.86, -95.15, -74.34),
    "BC": (48.30, 60.00, -139.05, -114.03),
    "AB": (49.00, 60.00, -120.00, -110.00),
    "MB": (49.00, 60.00, -102.00,  -89.00),
    "SK": (49.00, 60.00, -110.00,  -101.37),
    "NS": (43.37,  47.03, -66.32,  -59.68),
    "NB": (44.60,  48.07, -69.06,  -63.77),
}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def bbox_center_and_radius(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> tuple[float, float, float]:
    """
    Compute the centre of a bounding box and the radius (km) needed to
    cover all four corners. Used to summarise a province search in the
    API response.
    """
    clat = (lat_min + lat_max) / 2
    clon = (lon_min + lon_max) / 2
    corner_dist = haversine(clat, clon, lat_min, lon_min)
    return clat, clon, corner_dist


def filter_by_radius(
    companies: list[dict],
    lat: float,
    lon: float,
    radius_km: float,
) -> list[dict]:
    """Return companies within radius_km of (lat, lon), with distance attached."""
    results = []
    for c in companies:
        dist = haversine(lat, lon, c["lat"], c["lon"])
        if dist <= radius_km:
            results.append({**c, "_distance_km": round(dist, 2)})
    return results


def filter_by_province(
    companies: list[dict],
    province_code: str,
) -> tuple[list[dict], Optional[tuple[float, float, float]]]:
    """
    Return companies whose province field matches province_code.

    Also returns (centre_lat, centre_lon, covering_radius_km) for the
    known bounding box, or None if the province is not in PROVINCE_BBOXES.
    """
    matched = [c for c in companies if c.get("province", "").upper() == province_code.upper()]
    bbox = PROVINCE_BBOXES.get(province_code.upper())
    geo_info = bbox_center_and_radius(*bbox) if bbox else None
    return matched, geo_info
