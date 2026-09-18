"""Deterministic free-text metric extraction and topic detection."""

import re

from config import CATEGORICAL_METRICS, THRESHOLDS, VALID_RANGES

WINDOW_CHARS = 15

NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\d.])")

CATEGORY_PHRASES: dict[str, dict[str, list[str]]] = {
    "species_richness_trend": {
        "declining": [
            "declining",
            "decline",
            "decreasing",
            "losing species",
            "species loss",
            "worsening",
        ],
        "stable": ["stable", "unchanged", "holding steady"],
        "improving": ["improving", "increasing", "recovering", "more species"],
    },
    "land_use_category": {
        "monoculture": ["monoculture", "monocrop", "only wheat", "single crop", "just one crop"],
        "intercropped": ["intercropp", "intercropped", "intercropping", "inter-cropp"],
        "agroforestry": ["agroforest", "agroforestry", "agro-forest"],
        "natural_forest": ["natural forest", "native forest", "primary forest"],
    },
}

NUMERIC_RULES: dict[str, dict[str, object]] = {
    "soil_organic_carbon_pct": {
        "keywords": [
            "soil organic carbon",
            "soil carbon",
            "soc",
            "organic carbon",
            "organic matter",
        ],
        "units": ["%", "percent", "per cent", "pct"],
        "unit_required": False,
    },
    "rainfall_mm_year": {
        "keywords": ["rainfall", "rain", "precipitation", "annual rain"],
        "units": [
            "millimeters per year",
            "millimetres per year",
            "mm/year",
            "mm/yr",
            "mm",
            "millimeter",
            "millimeters",
            "millimetre",
            "millimetres",
            "per year",
        ],
        "unit_required": False,
    },
    "soil_ph": {
        "keywords": ["soil ph", "ph", "acidity"],
        "units": [],
        "unit_required": False,
    },
    "temperature_deviation_c": {
        "keywords": [
            "temperature deviation",
            "temp deviation",
            "warmer by",
            "degrees above",
        ],
        "units": ["°c", "c", "degrees celsius", "degree celsius", "degrees", "deg", "celsius"],
        "unit_required": False,
    },
    "ground_cover_pct": {
        "keywords": ["ground cover", "cover crop cover"],
        "units": ["%", "percent", "per cent", "pct"],
        "unit_required": True,
    },
}

_missing_numeric_rules = set(THRESHOLDS) - set(NUMERIC_RULES)
if _missing_numeric_rules:
    raise RuntimeError(
        f"NUMERIC_RULES is missing entries for: {sorted(_missing_numeric_rules)}",
    )

UNIT_TOKENS = {
    "%",
    "°c",
    "c",
    "mm",
    "millimeter",
    "millimeters",
    "millimetre",
    "millimetres",
    "mm/year",
    "mm/yr",
    "degrees",
    "deg",
    "celsius",
    "percent",
    "per cent",
    "pct",
}

TOPIC_PHRASES: dict[str, list[str]] = {
    "soil": [
        "soil",
        "bare soil",
        "carbon",
        "organic matter",
        "ph",
        "acidity",
        "erosion",
        "degraded soil",
        "fertility",
    ],
    "water_climate": [
        "rain",
        "rains",
        "rainfall",
        "precipitation",
        "drought",
        "arid",
        "dry",
        "water",
        "moisture",
        "temperature",
        "heat",
        "warmer",
        "cooler by",
        "degrees below",
        "degrees above",
        "colder",
    ],
    "biodiversity": [
        "biodiversity",
        "species",
        "wildlife",
        "pollinator",
        "habitat",
        "ecosystem",
        "flora",
        "fauna",
        "declining species",
    ],
    "land_use": [
        "monoculture",
        "crop",
        "farm",
        "land use",
        "field",
        "pasture",
        "forest",
        "agroforestry",
        "intercrop",
        "rotation",
    ],
}

# Contextual qualities that inform phrasing but are not metrics.
# Descriptors never enter the metrics dict and never affect the core-metric check.
# Vocabulary is drawn from the descriptive language already used in the corpus
# (FAO / USDA NRCS / USDA Forest Service documents on fragmentation, connectivity,
# degradation, dryland systems, erosion, waterways, and perennial cover).
DESCRIPTOR_PHRASES: dict[str, list[str]] = {
    "fragmentation": [
        "fragmentation",
        "fragmented",
        "patchy habitat",
        "isolated habitat",
        "isolated habitats",
    ],
    "connectivity": [
        "corridor",
        "corridors",
        "connectivity",
        "reconnect",
        "link habitat",
        "link habitats",
        "buffer strip",
        "buffer strips",
        "hedgerow",
        "hedgerows",
    ],
    "degradation": [
        "degraded",
        "depleted",
        "poor soil",
        "unproductive",
        "declining soil",
        "compacted",
        "compaction",
    ],
    "water_limited": [
        "semi-arid",
        "semi arid",
        "dryland",
        "dry land",
        "arid",
        "low rainfall",
        "low seasonal rainfall",
        "water scarcity",
        "water-limited",
        "water limited",
    ],
    "erosion_risk": [
        "erosion",
        "erosion-prone",
        "erosion prone",
        "runoff",
        "gullied",
        "exposed soil",
        "siltation",
    ],
    "slope": [
        "slope",
        "sloped",
        "hillside",
        "gradient",
    ],
    "waterways": [
        "stream",
        "streams",
        "river",
        "rivers",
        "riparian",
        "waterway",
        "waterways",
        "wetland",
        "wetlands",
        "downstream",
        "coastal",
        "seawater",
    ],
    "perennial_cover": [
        "perennial",
        "perennials",
        "tree cover",
        "shrub",
        "shrubs",
        "windbreak",
        "windbreaks",
        "orchard",
    ],
    "salinity": [
        "salinity",
        "saline",
        "high soil salinity",
        "salinization",
        "salinisation",
    ],
}


def extract_metrics(text: str) -> dict[str, float | str]:
    """
    Extract high-confidence metric values from free text.

    Structured inputs should still be preferred by callers; this function only
    returns values that are directly supported by nearby words and numbers.
    """
    extracted: dict[str, float | str] = {}
    if not text:
        return extracted

    normalized = text.lower()
    extracted.update(_extract_categories(normalized))
    extracted.update(_extract_numeric_metrics(normalized))
    return extracted


def detect_topics(text: str) -> set[str]:
    """Detect broad metric domains mentioned in free text."""
    if not text:
        return set()

    normalized = text.lower()
    topics: set[str] = set()
    for topic, phrases in TOPIC_PHRASES.items():
        if any(_phrase_in_text(normalized, phrase) for phrase in phrases):
            topics.add(topic)
    return topics


def detect_descriptors(text: str) -> set[str]:
    """
    Detect contextual descriptors mentioned in free text.

    Descriptors inform clarifying-question phrasing only. They are never treated
    as metrics and never affect the core-metric check.
    """
    if not text:
        return set()

    normalized = text.lower()
    descriptors: set[str] = set()
    for descriptor, phrases in DESCRIPTOR_PHRASES.items():
        if any(_phrase_in_text(normalized, phrase) for phrase in phrases):
            descriptors.add(descriptor)
    return descriptors


def _extract_categories(text: str) -> dict[str, str]:
    """Extract categorical metrics from closed-vocabulary phrase maps."""
    extracted: dict[str, str] = {}
    for metric_name, categories in CATEGORY_PHRASES.items():
        allowed_values = CATEGORICAL_METRICS.get(metric_name, [])
        matched_categories: list[str] = []
        for category, phrases in categories.items():
            if category not in allowed_values:
                continue
            if any(_phrase_in_text(text, phrase) for phrase in phrases):
                matched_categories.append(category)
        if len(matched_categories) == 1:
            extracted[metric_name] = matched_categories[0]
    return extracted


def _extract_numeric_metrics(text: str) -> dict[str, float]:
    """Extract numeric metric values when a keyword appears close to one number."""
    extracted: dict[str, float] = {}
    number_matches = [
        match for match in NUMBER_PATTERN.finditer(text) if not _is_range_part(text, match)
    ]

    for metric_name in THRESHOLDS:
        rule = NUMERIC_RULES.get(metric_name)
        if rule is None:
            continue

        candidates: list[float] = []
        for match in number_matches:
            value = _parse_number(match.group())
            if value is None or not _metric_keyword_near_number(text, match, rule):
                continue
            # Deviation is treated as a magnitude; negative deviations are not currently supported.
            if _value_in_valid_range(metric_name, value):
                candidates.append(value)

        if len(candidates) == 1:
            extracted[metric_name] = candidates[0]

    return extracted


def _phrase_in_text(text: str, phrase: str) -> bool:
    """Match phrases case-insensitively with word boundaries around word characters."""
    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _parse_number(raw: str) -> float | None:
    """Parse one integer or decimal token."""
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _metric_keyword_near_number(text: str, match: re.Match[str], rule: dict[str, object]) -> bool:
    """Check whether the number has a nearby metric keyword and acceptable unit."""
    keywords = rule["keywords"]
    if not isinstance(keywords, list) or not keywords:
        return False

    longest_keyword = max(len(str(keyword)) for keyword in keywords)
    window_chars = max(WINDOW_CHARS, longest_keyword + WINDOW_CHARS)

    start = max(0, match.start() - window_chars)
    end = min(len(text), match.end() + window_chars)
    before = text[start : match.start()]
    after = text[match.end() : end]
    window = f"{before}{match.group()}{after}"

    if not any(_phrase_in_text(window, str(keyword)) for keyword in keywords):
        return False

    units = rule["units"]
    unit_required = bool(rule["unit_required"])
    if not isinstance(units, list) or not units:
        return True

    has_unit = any(_unit_near_number(before, after, str(unit)) for unit in units)
    if not has_unit and _has_adjacent_unit(before, after):
        return False
    return has_unit or not unit_required


def _unit_near_number(before: str, after: str, unit: str) -> bool:
    """Return whether a unit appears directly beside a numeric token."""
    escaped = re.escape(unit)
    return (
        re.search(rf"\s*{escaped}(?!\w)", after, flags=re.IGNORECASE) is not None
        or re.search(rf"(?<!\w){escaped}\s*$", before, flags=re.IGNORECASE) is not None
    )


def _has_adjacent_unit(before: str, after: str) -> bool:
    """Detect an explicit adjacent unit so the wrong unit can be rejected."""
    return any(_unit_near_number(before, after, unit) for unit in UNIT_TOKENS)


def _is_range_part(text: str, match: re.Match[str]) -> bool:
    """Reject both numbers in compact ranges such as 350-400mm."""
    before = text[: match.start()].rstrip()
    after = text[match.end() :].lstrip()
    follows_range_dash = after.startswith("-") and len(after) > 1 and after[1].isdigit()
    follows_number_dash = before.endswith("-") and len(before) > 1 and before[-2].isdigit()
    return follows_range_dash or follows_number_dash


def _value_in_valid_range(metric_name: str, value: float) -> bool:
    """Check extracted numbers against the configured physical range."""
    min_value, max_value = VALID_RANGES[metric_name]
    return min_value <= value <= max_value
