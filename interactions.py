"""Detect cross-variable ecological interactions from classified metrics."""

import logging
from collections.abc import Callable
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Rule(NamedTuple):
    """A cross-metric ecological interaction rule."""

    rule_id: str
    description: str
    condition: Callable[[dict], bool]
    insight: str
    recommend_tags: list[str]


def _band(classifications: dict, metric_name: str) -> str | None:
    """Return the band_label for a metric, or None if not classified."""
    result = classifications.get(metric_name)
    return result["band_label"] if result else None


RULES: list[Rule] = [
    Rule(
        rule_id="soil_water_compounding",
        description="Low soil carbon compounds dry rainfall stress.",
        condition=lambda c: _band(c, "soil_organic_carbon_pct") in {"critical", "low"}
        and _band(c, "rainfall_mm_year") in {"arid", "semi_arid"},
        insight=(
            "Low soil organic carbon can reduce water retention, intensifying dry-season stress."
        ),
        recommend_tags=["increase_soil_carbon", "mulch", "water_retention"],
    ),
    Rule(
        rule_id="monoculture_carbon_loss",
        description="Low land-use diversity coincides with depleted soil carbon.",
        condition=lambda c: _band(c, "land_use_category") == "monoculture"
        and _band(c, "soil_organic_carbon_pct") in {"critical", "low"},
        insight=(
            "Simplified vegetation structure can limit litter inputs and accelerate "
            "soil carbon decline."
        ),
        recommend_tags=["diversify_planting", "cover_crops", "agroforestry"],
    ),
    Rule(
        rule_id="water_species_stress",
        description="Dry conditions coincide with declining native species richness.",
        condition=lambda c: _band(c, "rainfall_mm_year") in {"arid", "semi_arid"}
        and _band(c, "species_richness_trend") == "declining",
        insight="Water scarcity may be narrowing habitat suitability for native species.",
        recommend_tags=["drought_refugia", "native_species_recovery", "water_harvesting"],
    ),
    Rule(
        rule_id="ph_carbon_nutrient_cycling",
        description="Soil pH stress overlaps with low soil carbon.",
        condition=lambda c: _band(c, "soil_ph") in {"acidic", "alkaline"}
        and _band(c, "soil_organic_carbon_pct") in {"critical", "low"},
        insight="pH imbalance and low carbon can slow nutrient cycling and biological activity.",
        recommend_tags=["soil_testing", "organic_amendments", "ph_management"],
    ),
    Rule(
        rule_id="dryland_habitat_fragmentation",
        description=(
            "Dry, simplified landscapes with low cover suggest habitat fragmentation pressure."
        ),
        condition=lambda c: _band(c, "rainfall_mm_year") in {"arid", "semi_arid"}
        and _band(c, "land_use_category") == "monoculture"
        and _band(c, "ground_cover_pct") in {"low_cover", "partial_cover"},
        insight="Dryness, low ground cover, and simplified land use can separate habitat patches.",
        recommend_tags=["habitat_corridors", "ground_cover", "native_hedgerows"],
    ),
    Rule(
        rule_id="heat_drought_compounding_stress",
        description="Temperature deviation compounds dry rainfall stress.",
        condition=lambda c: _band(c, "temperature_deviation_c")
        in {"moderate_stress", "high_stress"}
        and _band(c, "rainfall_mm_year") in {"arid", "semi_arid"},
        insight="Heat stress can raise water demand while rainfall is already limiting.",
        recommend_tags=["shade", "drought_tolerant_species", "soil_moisture"],
    ),
    Rule(
        rule_id="unbuffered_erosion_risk",
        description="Bare ground and low soil carbon leave soil vulnerable to erosion.",
        condition=lambda c: _band(c, "ground_cover_pct") in {"low_cover", "partial_cover"}
        and _band(c, "soil_organic_carbon_pct") in {"critical", "low"},
        insight="Low cover and weak soil structure can leave the site exposed to erosion losses.",
        recommend_tags=["ground_cover", "contour_barriers", "soil_structure"],
    ),
    Rule(
        rule_id="good_practice_reinforcement",
        description="Healthy carbon, cover, and diversity indicate resilient management.",
        condition=lambda c: _band(c, "soil_organic_carbon_pct") == "good"
        and _band(c, "ground_cover_pct") in {"high_cover", "very_high_cover"}
        and _band(c, "land_use_category") in {"intercropped", "agroforestry", "natural_forest"}
        and _band(c, "soil_ph") in {"slightly_acidic", "neutral"},
        insight=(
            "Healthy soil carbon, protective cover, and diverse land use support system resilience."
        ),
        recommend_tags=["maintain_cover", "protect_diversity", "monitor_soil_health"],
    ),
]


def detect_interactions(classifications: dict) -> list[Rule]:
    """Return every rule whose condition is satisfied by the classifications."""
    matches: list[Rule] = []
    for rule in RULES:
        try:
            if rule.condition(classifications):
                matches.append(rule)
        except (KeyError, TypeError):
            logger.exception("Skipping rule with invalid condition: %s", rule.rule_id)
    return matches
