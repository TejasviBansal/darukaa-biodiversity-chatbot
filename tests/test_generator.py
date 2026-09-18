"""Tests for prompt and situation-summary construction."""

from generator import (
    DESCRIPTOR_SUMMARY_PHRASINGS,
    OUTPUT_SCHEMA_INSTRUCTIONS,
    build_prompt,
    build_situation_summary,
)
from interactions import Rule


def test_build_situation_summary_includes_metrics_and_interactions() -> None:
    """The summary includes classified metric bands and detected rule descriptions."""
    classifications = {
        "soil_organic_carbon_pct": {
            "metric_name": "soil_organic_carbon_pct",
            "value": 0.3,
            "band_label": "critical",
            "source": "FAO GSOCmap guidance",
        },
        "land_use_category": {
            "metric_name": "land_use_category",
            "value": "monoculture",
            "band_label": "monoculture",
            "source": "IPCC AFOLU land-use categories and FAO agroforestry overview",
        },
    }
    rules = [
        Rule(
            rule_id="soil_water_compounding",
            description="Low soil carbon compounds dry rainfall stress.",
            condition=lambda _: True,
            insight="Low soil organic carbon can reduce water retention.",
            recommend_tags=["increase_soil_carbon"],
        ),
    ]

    summary = build_situation_summary(classifications, rules)

    assert "soil_organic_carbon_pct: critical (0.3)." in summary
    assert "land_use_category: monoculture." in summary
    assert "Low soil carbon compounds dry rainfall stress" in summary


def test_build_prompt_includes_summary_sources_context_history_and_schema() -> None:
    """The prompt exposes exact source names and the required JSON schema."""
    situation_summary = "soil_organic_carbon_pct: critical (0.3)."
    retrieved_chunks = [
        {
            "chunk_id": "cover_chunk_0",
            "text": "Cover crops add biomass and protect soil.",
            "intervention": "cover_cropping",
            "metric": "soil_organic_carbon",
            "source": "FAO Soil Organic Cover in Conservation Agriculture",
            "source_url": "https://example.com/source",
            "region": "general",
            "similarity_score": 0.61,
        },
    ]
    history = [{"role": "user", "content": "What should I do first?"}]

    prompt = build_prompt(situation_summary, retrieved_chunks, history)

    assert situation_summary in prompt
    assert "SOURCE: FAO Soil Organic Cover in Conservation Agriculture" in prompt
    assert "Cover crops add biomass and protect soil." in prompt
    assert "user: What should I do first?" in prompt
    assert OUTPUT_SCHEMA_INSTRUCTIONS in prompt


def test_prompt_instructs_priority_for_quantitative_estimates() -> None:
    """The prompt must instruct the model to prioritize numeric estimates from evidence."""
    lowered = OUTPUT_SCHEMA_INSTRUCTIONS.lower()
    assert "prioritize" in lowered
    assert "quantitative" in lowered
    assert "do not invent" in lowered


def test_prompt_instructs_naming_specific_species_and_practices() -> None:
    """The prompt must instruct the model to name species and practices from evidence."""
    lowered = OUTPUT_SCHEMA_INSTRUCTIONS.lower()
    assert "species" in lowered
    assert "functional group" in lowered or "practice" in lowered
    assert "only use names that appear" in lowered


def test_build_prompt_includes_situation_and_context() -> None:
    """build_prompt includes the situation summary and retrieved chunk sources."""
    chunks = [
        {
            "source": "Test Source",
            "intervention": "cover_cropping",
            "metric": "soil_organic_carbon",
            "region": "general",
            "similarity_score": 0.5,
            "text": "Cover crops can raise soil organic carbon by 0.2-0.5 percentage points.",
        },
    ]
    prompt = build_prompt("soil_organic_carbon_pct: critical (0.3).", chunks, [])
    assert "soil_organic_carbon_pct: critical (0.3)." in prompt
    assert "Test Source" in prompt
    assert "0.2-0.5 percentage points" in prompt
    assert "prioritize" in prompt.lower()


def test_build_situation_summary_appends_descriptors_when_provided() -> None:
    """Descriptors, when provided, appear in the situation summary."""
    classifications = {
        "soil_organic_carbon_pct": {
            "metric_name": "soil_organic_carbon_pct",
            "value": 0.3,
            "band_label": "critical",
            "source": "FAO GSOCmap guidance",
        },
    }
    summary = build_situation_summary(
        classifications,
        [],
        descriptors={"fragmentation", "connectivity"},
    )
    assert "User context descriptors:" in summary
    assert "habitat fragmentation" in summary
    assert "connectivity and corridors" in summary


def test_build_situation_summary_omits_descriptor_line_when_empty() -> None:
    """No descriptor line appears when descriptors are empty or absent."""
    classifications = {
        "soil_organic_carbon_pct": {
            "metric_name": "soil_organic_carbon_pct",
            "value": 0.3,
            "band_label": "critical",
            "source": "FAO GSOCmap guidance",
        },
    }
    summary_default = build_situation_summary(classifications, [])
    summary_empty = build_situation_summary(classifications, [], descriptors=set())
    assert "User context descriptors:" not in summary_default
    assert "User context descriptors:" not in summary_empty


def test_build_situation_summary_backward_compatible() -> None:
    """The old two-argument call still works and produces a valid summary."""
    classifications = {
        "rainfall_mm_year": {
            "metric_name": "rainfall_mm_year",
            "value": 350.0,
            "band_label": "semi_arid",
            "source": "UNEP Aridity Index classification",
        },
    }
    summary = build_situation_summary(classifications, [])
    assert "rainfall_mm_year: semi_arid (350.0)." in summary
    assert "Detected interactions: none." in summary


def test_descriptor_summary_phrasings_cover_all_descriptors() -> None:
    """Every descriptor key used by extractor has a summary phrasing."""
    from extractor import DESCRIPTOR_PHRASES

    missing = set(DESCRIPTOR_PHRASES) - set(DESCRIPTOR_SUMMARY_PHRASINGS)
    assert not missing, f"Missing summary phrasings for: {sorted(missing)}"
