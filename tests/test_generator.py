"""Tests for prompt and situation-summary construction."""

from generator import OUTPUT_SCHEMA_INSTRUCTIONS, build_prompt, build_situation_summary
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
