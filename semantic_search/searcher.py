"""
Semantic search with geo-filtering.

Search flow:
  1. Parse the query to extract the semantic part, location, and radius.
  2. Apply geographic pre-filtering:
       - Province code (e.g. "QC") -> bounding-box filter
       - City name                 -> Haversine radius filter (requires geocoding)
       - No location               -> all companies are candidates
  3. Score remaining candidates with cosine similarity (FAISS inner product
     on L2-normalised embeddings ≡ cosine similarity).
  4. Return results sorted by score descending.

Note: For the example we skip live geocoding of city names and instead rely
on a small lookup table. In production, use geopy's Nominatim with caching
and rate-limiting to resolve arbitrary location strings to coordinates.
"""
import numpy as np
from geo_utils import PROVINCE_BBOXES, filter_by_province, filter_by_radius
from sentence_transformers import SentenceTransformer

# Minimal city to (lat, lon) lookup for the example dataset.
# In production, replace this with a geocoding API call.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "montréal":      (45.5017, -73.5673),
    "montreal":      (45.5017, -73.5673),
    "laval":         (45.5881, -73.6917),
    "québec":        (46.8139, -71.2082),
    "quebec":        (46.8139, -71.2082),
    "sherbrooke":    (45.4042, -71.8929),
    "longueuil":     (45.5313, -73.5185),
    "drummondville": (45.8833, -72.4833),
    "trois-rivières":(46.3432, -72.5450),
    "trois-rivieres":(46.3432, -72.5450),
    "toronto":       (43.6532, -79.3832),
    "ottawa":        (45.4215, -75.6972),
}


def search(
    query: str,
    location: str | None,
    radius_km: float,
    companies: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 10,
) -> list[dict]:
    """
    Execute a semantic + geo search.

    Args:
        query:      Semantic part of the query (already stripped of location/radius).
        location:   Location string or None.
        radius_km:  Search radius in km (used for city-level searches).
        companies:  Company metadata list (parallel to embeddings).
        embeddings: (N, D) float32 array of L2-normalised embeddings.
        model:      SentenceTransformer instance used to encode the query.
        top_k:      Maximum results to return.

    Returns:
        List of result dicts with score, distance_km, and company fields.
    """
    candidates = list(range(len(companies)))
    geo_info = None
    strategy = "none"

    if location:
        loc_lower = location.lower().strip()

        # ── Province-level: bounding box ──────────────────────────────────────
        if loc_lower.upper() in PROVINCE_BBOXES or loc_lower.upper() in {p.lower() for p in PROVINCE_BBOXES}:
            code = loc_lower.upper()
            filtered, geo_info = filter_by_province(companies, code)
            candidates = [i for i, c in enumerate(companies) if c in filtered]
            strategy = f"province bounding-box ({code})"

        # ── City-level: Haversine radius ──────────────────────────────────────
        elif loc_lower in CITY_COORDS:
            lat, lon = CITY_COORDS[loc_lower]
            filtered = filter_by_radius(companies, lat, lon, radius_km)
            filtered_ids = {c["id"] for c in filtered}
            candidates = [i for i, c in enumerate(companies) if c["id"] in filtered_ids]
            strategy = f"city radius ({location}, {radius_km} km)"

        else:
            print(f"[geo] Location '{location}' not found in lookup — returning all candidates.")

    print(f"[search] Strategy: {strategy}")
    print(f"[search] Candidates after geo-filter: {len(candidates)}")

    if not candidates:
        return []

    # ── Semantic scoring ──────────────────────────────────────────────────────
    q_vec = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    candidate_embs = embeddings[candidates]
    scores = candidate_embs @ q_vec  # cosine similarity for normalised vecs

    # Sort by score descending
    order = np.argsort(-scores)
    results = []
    for rank in order[:top_k]:
        idx = candidates[rank]
        c = companies[idx]
        result = {
            "rank":         int(rank) + 1,
            "score":        round(float(scores[rank]), 4),
            "name":         c["name"],
            "city":         c["city"],
            "province":     c["province"],
            "capabilities": c.get("capabilities", []),
            "certifications": c.get("certifications", []),
        }
        if "_distance_km" in c:
            result["distance_km"] = c["_distance_km"]
        results.append(result)

    return results
