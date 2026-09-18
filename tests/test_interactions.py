"""Tests for ecological interaction detection."""

from classifier import classify_all
from interactions import detect_interactions


def test_bad_scenario_fires_expected_risk_rules() -> None:
    """Low carbon, dry rainfall, monoculture, low cover, and declining richness fire risks."""
    classifications = classify_all(
        {
            "soil_organic_carbon_pct": 0.4,
            "rainfall_mm_year": 200.0,
            "land_use_category": "monoculture",
            "species_richness_trend": "declining",
            "ground_cover_pct": 20.0,
            "soil_ph": 8.0,
            "temperature_deviation_c": 3.0,
        },
    )

    rule_ids = {rule.rule_id for rule in detect_interactions(classifications)}

    assert "soil_water_compounding" in rule_ids
    assert "monoculture_carbon_loss" in rule_ids
    assert "water_species_stress" in rule_ids
    assert "dryland_habitat_fragmentation" in rule_ids
    assert "good_practice_reinforcement" not in rule_ids


def test_good_scenario_fires_positive_rule_only() -> None:
    """Healthy carbon, cover, diversity, pH, rainfall, and temperature avoid risk rules."""
    classifications = classify_all(
        {
            "soil_organic_carbon_pct": 3.0,
            "rainfall_mm_year": 900.0,
            "land_use_category": "agroforestry",
            "species_richness_trend": "improving",
            "ground_cover_pct": 90.0,
            "soil_ph": 7.0,
            "temperature_deviation_c": 1.0,
        },
    )

    rules = detect_interactions(classifications)
    rule_ids = {rule.rule_id for rule in rules}

    assert rule_ids == {"good_practice_reinforcement"}
