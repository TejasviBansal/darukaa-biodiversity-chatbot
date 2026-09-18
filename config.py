"""All application configuration and scientific thresholds in one place."""

import os
from typing import NamedTuple

# --- App settings, env-driven with defaults ---

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "biodiversity_knowledge")
CORPUS_ROOT_DIR = os.getenv("CORPUS_ROOT_DIR", "./corpus")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
RETRIEVAL_SIMILARITY_CUTOFF = float(os.getenv("RETRIEVAL_SIMILARITY_CUTOFF", "0.42"))
MAX_FINAL_CHUNKS = int(os.getenv("MAX_FINAL_CHUNKS", "5"))
SESSION_HISTORY_MAX_TURNS = int(os.getenv("SESSION_HISTORY_MAX_TURNS", "10"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Scientific thresholds ---
# Each band: (label, min_value, max_value, source)
# Bands are lower-inclusive, upper-exclusive, except the last band per metric
# which is inclusive on both ends (so the maximum valid value classifies correctly).


class Band(NamedTuple):
    """A named scientific threshold band for one metric."""

    label: str
    min_value: float
    max_value: float
    source: str


THRESHOLDS: dict[str, list[Band]] = {
    "soil_organic_carbon_pct": [
        Band("critical", 0.0, 0.5, "FAO GSOCmap guidance"),
        Band("low", 0.5, 1.0, "FAO GSOCmap guidance"),
        Band("moderate", 1.0, 2.0, "FAO GSOCmap guidance"),
        Band("good", 2.0, float("inf"), "FAO GSOCmap guidance"),
    ],
    "rainfall_mm_year": [
        Band("arid", 0.0, 250.0, "UNEP Aridity Index classification"),
        Band("semi_arid", 250.0, 500.0, "UNEP Aridity Index classification"),
        Band("sub_humid", 500.0, 1000.0, "UNEP Aridity Index classification"),
        Band("humid", 1000.0, float("inf"), "UNEP Aridity Index classification"),
    ],
    "soil_ph": [
        Band("acidic", 0.0, 5.5, "USDA NRCS Soil Quality Test Kit Guide"),
        Band("slightly_acidic", 5.5, 6.5, "USDA NRCS Soil Quality Test Kit Guide"),
        Band("neutral", 6.5, 7.5, "USDA NRCS Soil Quality Test Kit Guide"),
        Band("alkaline", 7.5, 14.0, "USDA NRCS Soil Quality Test Kit Guide"),
    ],
    "temperature_deviation_c": [
        Band("optimal", 0.0, 2.0, "FAO Ecocrop climatic suitability framework"),
        Band("moderate_stress", 2.0, 4.0, "FAO Ecocrop climatic suitability framework"),
        Band("high_stress", 4.0, float("inf"), "FAO Ecocrop climatic suitability framework"),
    ],
    "ground_cover_pct": [
        Band("low_cover", 0.0, 30.0, "FAO Conservation Agriculture principles"),
        Band("partial_cover", 30.0, 60.0, "FAO Conservation Agriculture principles"),
        Band("high_cover", 60.0, 90.0, "FAO Conservation Agriculture principles"),
        Band("very_high_cover", 90.0, 100.0, "FAO Conservation Agriculture principles"),
    ],
}

# --- Categorical ecological inputs ---

LAND_USE_CATEGORIES: list[str] = [
    "monoculture",
    "intercropped",
    "agroforestry",
    "natural_forest",
]

SPECIES_RICHNESS_TRENDS: list[str] = [
    "declining",
    "stable",
    "improving",
]

CATEGORICAL_METRICS: dict[str, list[str]] = {
    "land_use_category": LAND_USE_CATEGORIES,
    "species_richness_trend": SPECIES_RICHNESS_TRENDS,
}

CATEGORICAL_SOURCES: dict[str, str] = {
    "land_use_category": "IPCC AFOLU land-use categories and FAO agroforestry overview",
    "species_richness_trend": "Relative biodiversity trend assessment",
}

# Physically valid ranges for input validation - same keys as THRESHOLDS
VALID_RANGES: dict[str, tuple[float, float]] = {
    "soil_organic_carbon_pct": (0.0, 100.0),
    "rainfall_mm_year": (0.0, float("inf")),
    "soil_ph": (0.0, 14.0),
    "temperature_deviation_c": (0.0, float("inf")),
    "ground_cover_pct": (0.0, 100.0),
}
