"""Tests for deterministic free-text metric extraction."""

from extractor import detect_descriptors, detect_topics, extract_metrics


def test_extracts_species_richness_trend() -> None:
    """Known biodiversity trend phrases map to configured categories."""
    assert extract_metrics("biodiversity is declining on my land") == {
        "species_richness_trend": "declining",
    }


def test_extracts_monoculture_land_use() -> None:
    """Known monoculture phrasing maps to the land-use category."""
    assert extract_metrics("I grow only wheat") == {"land_use_category": "monoculture"}


def test_extracts_agroforestry_land_use() -> None:
    """Known agroforestry phrasing maps to the land-use category."""
    assert extract_metrics("we practice agroforestry") == {"land_use_category": "agroforestry"}


def test_rejects_conflicting_categorical_phrases() -> None:
    """Multiple conflicting trend words should extract nothing, not the first match."""
    assert "species_richness_trend" not in extract_metrics(
        "biodiversity is stable, species are declining",
    )


def test_extracts_soil_organic_carbon_number() -> None:
    """SOC values are extracted when a nearby soil-carbon keyword is present."""
    assert extract_metrics("soil carbon is 0.3%") == {"soil_organic_carbon_pct": 0.3}


def test_extracts_spelled_out_percent_unit() -> None:
    """Spelled-out units should extract the same as symbol units."""
    assert extract_metrics("soil carbon 0.3 percent") == {"soil_organic_carbon_pct": 0.3}


def test_extracts_long_multiword_keyword_with_separator() -> None:
    """Multi-word keywords must be reachable even with a short separator before the number."""
    assert extract_metrics("soil organic carbon at 0.35%") == {
        "soil_organic_carbon_pct": 0.35,
    }
    assert extract_metrics("annual rainfall is about 350 mm") == {"rainfall_mm_year": 350.0}


def test_rejects_wrong_explicit_unit() -> None:
    """A number with the wrong unit should not extract for that metric."""
    assert "soil_organic_carbon_pct" not in extract_metrics("soil carbon 0.3 mm")


def test_extracts_rainfall_number() -> None:
    """Rainfall values are extracted with nearby rainfall words and mm units."""
    assert extract_metrics("rainfall 350 mm/year") == {"rainfall_mm_year": 350.0}


def test_extracts_soil_ph_number() -> None:
    """Soil pH values are extracted when in range."""
    assert extract_metrics("soil pH is 6.8") == {"soil_ph": 6.8}


def test_does_not_extract_unrelated_counts() -> None:
    """Unrelated counts should not become numeric metrics."""
    extracted = extract_metrics("I have 3 fields")

    assert "soil_organic_carbon_pct" not in extracted
    assert not any(isinstance(value, float) for value in extracted.values())


def test_rejects_numeric_ranges() -> None:
    """Compact ranges are deliberately skipped rather than averaged."""
    assert "rainfall_mm_year" not in extract_metrics("rainfall 350-400mm")


def test_rejects_out_of_range_values() -> None:
    """Values outside configured valid ranges are skipped."""
    assert "soil_ph" not in extract_metrics("soil pH 20")


def test_temperature_deviation_extracts_magnitude() -> None:
    """Temperature deviations extract as positive magnitudes regardless of direction."""
    assert extract_metrics("warmer by 2 degrees") == {"temperature_deviation_c": 2.0}
    assert "temperature_deviation_c" not in extract_metrics("cooler by 2 degrees")


def test_detects_biodiversity_topic_only() -> None:
    """Biodiversity text should not imply numeric domains."""
    assert detect_topics("biodiversity is declining") == {"biodiversity"}


def test_detects_soil_and_water_topics() -> None:
    """Topic detection is intentionally broad for clarifying-question wording."""
    assert detect_topics("the soil is degraded and it barely rains") >= {
        "soil",
        "water_climate",
    }


def test_empty_topic_and_extraction_for_vague_text() -> None:
    """Vague text with no configured phrases returns no topics or metrics."""
    text = "nothing grows here anymore"

    assert detect_topics(text) == set()
    assert extract_metrics(text) == {}


def test_bare_soil_mentions_detect_soil_topic_without_extracting_numeric() -> None:
    """'Bare soil' is a topic signal, not a numeric value."""
    assert "ground_cover_pct" not in extract_metrics("the field is bare soil")
    assert "soil" in detect_topics("the field is bare soil")


def test_directional_temperature_phrases_detect_water_climate_topic() -> None:
    """Directional temperature phrasing should trigger topic detection even without extraction."""
    assert "water_climate" in detect_topics("cooler by 2 degrees this year")
    assert "water_climate" in detect_topics("temperatures are degrees below normal")
    assert "temperature_deviation_c" not in extract_metrics("cooler by 2 degrees this year")


def test_detects_fragmentation_and_connectivity_descriptors() -> None:
    """Fragmentation and connectivity language should be detected as descriptors."""
    text = (
        "habitat fragmentation is dropping native pollinator diversity; "
        "what interventions will reconnect these corridors?"
    )
    descriptors = detect_descriptors(text)
    assert "fragmentation" in descriptors
    assert "connectivity" in descriptors


def test_detects_water_limited_descriptor() -> None:
    """Semi-arid and dryland phrasing should be detected as water_limited."""
    assert "water_limited" in detect_descriptors("semi-arid zone with low seasonal rainfall")
    assert "water_limited" in detect_descriptors("dryland wheat system")


def test_detects_erosion_slope_waterways_descriptors() -> None:
    """Erosion, slope, and waterways vocabulary should be detected."""
    text = (
        "Heavy seasonal rainfall on newly deforested, sloped red-loam soils is "
        "causing massive topsoil erosion and downstream siltation."
    )
    descriptors = detect_descriptors(text)
    assert "erosion_risk" in descriptors
    assert "slope" in descriptors
    assert "waterways" in descriptors


def test_detects_salinity_descriptor() -> None:
    """Salinity vocabulary should be detected as its own descriptor."""
    text = "Coastal farmlands are experiencing seawater intrusion and high soil salinity."
    descriptors = detect_descriptors(text)
    assert "salinity" in descriptors
    assert "waterways" in descriptors


def test_descriptors_never_become_metrics() -> None:
    """Descriptor vocabulary must never appear in extract_metrics output."""
    text = (
        "Our tea plantation borders a tropical reserve, but habitat fragmentation "
        "is dropping native pollinator and avian diversity."
    )
    assert extract_metrics(text) == {}
    descriptors = detect_descriptors(text)
    assert "fragmentation" in descriptors


def test_detect_descriptors_returns_empty_for_plain_text() -> None:
    """Text with no descriptor vocabulary returns an empty set."""
    assert detect_descriptors("nothing notable here") == set()
