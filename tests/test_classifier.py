"""Tests for threshold classification."""

import math

import pytest

from classifier import classify_all, classify_metric


@pytest.mark.parametrize(
    ("metric_name", "cases"),
    [
        (
            "soil_organic_carbon_pct",
            [(0.0, "critical"), (0.5, "low"), (1.0, "moderate"), (2.0, "good")],
        ),
        (
            "rainfall_mm_year",
            [(0.0, "arid"), (250.0, "semi_arid"), (500.0, "sub_humid"), (1000.0, "humid")],
        ),
        (
            "soil_ph",
            [(0.0, "acidic"), (5.5, "slightly_acidic"), (6.5, "neutral"), (7.5, "alkaline")],
        ),
        (
            "temperature_deviation_c",
            [(0.0, "optimal"), (2.0, "moderate_stress"), (4.0, "high_stress")],
        ),
        (
            "ground_cover_pct",
            [
                (0.0, "low_cover"),
                (30.0, "partial_cover"),
                (60.0, "high_cover"),
                (90.0, "very_high_cover"),
            ],
        ),
    ],
)
def test_metric_boundaries_classify_correctly(
    metric_name: str,
    cases: list[tuple[float, str]],
) -> None:
    """Each metric classifies boundary values into the lower-inclusive band."""
    for value, expected_band in cases:
        assert classify_metric(metric_name, value)["band_label"] == expected_band


def test_classifier_rejects_invalid_values() -> None:
    """Invalid metric values raise clear errors."""
    with pytest.raises(ValueError, match="non-finite"):
        classify_metric("soil_ph", math.nan)

    with pytest.raises(ValueError, match="between"):
        classify_metric("soil_ph", 15.0)


def test_categorical_metrics_validate_known_values() -> None:
    """Categorical metrics are validated directly instead of threshold-binned."""
    land_use = classify_metric("land_use_category", "agroforestry")
    trend = classify_metric("species_richness_trend", "declining")

    assert land_use["band_label"] == "agroforestry"
    assert trend["band_label"] == "declining"


def test_categorical_metrics_reject_unknown_values() -> None:
    """Categorical metrics reject values outside their configured vocabulary."""
    with pytest.raises(ValueError, match="must be one of"):
        classify_metric("land_use_category", "orchard")


def test_classifier_rejects_unknown_metric_name() -> None:
    """Unknown metric names are rejected for single-metric classification."""
    with pytest.raises(ValueError, match="Unknown metric"):
        classify_metric("mystery_metric", 1.0)


def test_classify_all_skips_unknown_metrics(caplog: pytest.LogCaptureFixture) -> None:
    """Bulk classification skips unknown metrics with a warning."""
    result = classify_all({"soil_ph": 7.0, "mystery_metric": 1.0})

    assert set(result) == {"soil_ph"}
    assert "Skipping unknown metric" in caplog.text
