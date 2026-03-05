"""
Natural language query parser.

Extracts three components from a free-text query:
  - semantic_query : the manufacturing/capability part
  - location       : city or province name, if mentioned
  - radius_km      : numeric distance, if mentioned (default 50 km)

Supports English and French location prepositions.

Examples:
  "CNC machining aluminum in Montréal within 50 km"
      -> ("CNC machining aluminum", "Montréal", 50.0)
  "soudure TIG dans la région de Québec"
      -> ("soudure TIG", "Québec", 50.0)
  "laser cutting in QC"
      -> ("laser cutting", "QC", 50.0)
"""
import re

DEFAULT_RADIUS_KM = 50.0

# French prepositions → English equivalents (applied before pattern matching)
_FR_PREP = [
    (r"\bà\s+",                    "in "),
    (r"\bau\s+",                   "in "),
    (r"\baux\s+",                  "in "),
    (r"\ben\s+(?!\d)",             "in "),   # "en" only when not followed by a digit (avoids "EN 9100")
    (r"\bprès\s+de\s+",            "near "),
    (r"\bautour\s+de\s+",          "around "),
    (r"\bdepuis\s+",               "from "),
    (r"\bdans\s+la\s+région\s+de\s+", "in "),
    (r"\bdans\s+l[e']\s*",         "in "),
    (r"\bdans\s+",                 "in "),
]

_DIST_RE = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>km|kilometers?|kilometres?|miles?|mi)\b",
    re.IGNORECASE,
)

_LOC_RE = re.compile(
    r"\b(?:in|near|around|from|within)\s+([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\s,.\'-]{1,40}?)(?=\s*$|\s+\d|\s+(?:area|region|province|within|km|miles?))",
    re.IGNORECASE,
)


def _normalize_fr(text: str) -> str:
    for pattern, replacement in _FR_PREP:
        text = re.sub(pattern, replacement, text)
    return text


def parse_query(
    raw: str,
    default_radius: float = DEFAULT_RADIUS_KM,
) -> tuple[str, str | None, float]:
    """
    Parse a natural language query into its components.

    Returns:
        (semantic_query, location, radius_km)
        location is None if no location was detected.
    """
    text = _normalize_fr(raw)
    radius_km = default_radius
    location: str | None = None
    spans_to_remove: list[tuple[int, int]] = []

    # Extract distance
    dm = _DIST_RE.search(text)
    if dm:
        val = float(dm.group("val"))
        unit = dm.group("unit").lower()
        radius_km = val if unit.startswith("km") else val * 1.60934
        spans_to_remove.append(dm.span())

    # Extract location
    lm = _LOC_RE.search(text)
    if lm:
        location = lm.group(1).strip().rstrip(",.")
        spans_to_remove.append(lm.span())

    # Remove matched spans from right to left to preserve indices
    clean = text
    for start, end in sorted(spans_to_remove, reverse=True):
        clean = clean[:start] + clean[end:]

    # Strip residual prepositions left at the end, collapse whitespace
    clean = re.sub(r"\b(in|near|around|from|within)\s*$", "", clean, flags=re.IGNORECASE)
    clean = " ".join(clean.split())

    return clean, location, radius_km
