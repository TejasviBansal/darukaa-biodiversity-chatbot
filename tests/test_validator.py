"""Tests for evidence-grounding validation."""

from validator import (
    check_numbers_grounded,
    compute_confidence,
    validate_recommendations,
    verify_source,
)


def _chunk(source: str, text: str, similarity_score: float) -> dict:
    """Create a minimal retrieved chunk for validator tests."""
    return {
        "chunk_id": "chunk_0",
        "text": text,
        "intervention": "cover_cropping",
        "metric": "soil_organic_carbon",
        "source": source,
        "source_url": "https://example.com",
        "region": "general",
        "similarity_score": similarity_score,
    }


def test_matching_source_and_grounded_numbers_pass_with_confidence() -> None:
    """A source-matched recommendation with evidence-backed numbers survives."""
    chunks = [
        _chunk(
            "SARE Cover Crop Guidance",
            "Cover crops reduced soil loss by about 31-100 percent in field evidence.",
            0.58,
        ),
    ]
    llm_output = {
        "recommendations": [
            {
                "recommendation": "Use cover crops to reduce soil loss by 31-100 percent.",
                "mechanism": "Roots and residues protect bare soil.",
                "metrics_impacted": ["erosion_risk"],
                "time_horizon": "medium",
                "source": "SARE Cover Crop Guidance",
            },
        ],
    }

    result = validate_recommendations(llm_output, chunks)

    assert result["dropped_count"] == 0
    assert result["recommendations"][0]["confidence"] == "high"


def test_unmatched_source_is_dropped() -> None:
    """Recommendations citing a source outside retrieved evidence are removed."""
    chunks = [_chunk("Known Source", "Cover crops protect soil.", 0.58)]
    llm_output = {
        "recommendations": [
            {
                "recommendation": "Use cover crops.",
                "mechanism": "They protect soil.",
                "metrics_impacted": ["erosion_risk"],
                "time_horizon": "medium",
                "source": "Invented Source",
            },
        ],
    }

    result = validate_recommendations(llm_output, chunks)

    assert result == {"recommendations": [], "dropped_count": 1}


def test_ungrounded_number_is_dropped() -> None:
    """A number not present in the retrieved evidence triggers a drop."""
    chunks = [_chunk("Known Source", "Cover crops protect soil without a numeric claim.", 0.58)]
    llm_output = {
        "recommendations": [
            {
                "recommendation": "Use cover crops to reduce erosion by 42 percent.",
                "mechanism": "Residues protect soil.",
                "metrics_impacted": ["erosion_risk"],
                "time_horizon": "medium",
                "source": "Known Source",
            },
        ],
    }

    result = validate_recommendations(llm_output, chunks)

    assert result == {"recommendations": [], "dropped_count": 1}


def test_recommendation_with_no_numbers_passes_numeric_check() -> None:
    """Qualitative recommendations do not need numeric grounding."""
    chunks = [_chunk("Known Source", "Cover crops protect soil.", 0.5)]
    recommendation = {
        "recommendation": "Use cover crops to keep soil covered.",
        "mechanism": "Roots and residues reduce bare soil exposure.",
        "source": "Known Source",
    }

    assert check_numbers_grounded(recommendation, chunks)
    assert verify_source(recommendation, chunks)


def test_compute_confidence_tiers() -> None:
    """Confidence comes from retrieved similarity scores for matching sources."""
    recommendation = {"source": "Known Source"}

    assert compute_confidence(recommendation, [_chunk("Known Source", "Text", 0.55)]) == "high"
    assert compute_confidence(recommendation, [_chunk("Known Source", "Text", 0.45)]) == "medium"
    assert compute_confidence(recommendation, [_chunk("Known Source", "Text", 0.44)]) == "low"
    assert compute_confidence(recommendation, [_chunk("Other Source", "Text", 0.9)]) == "low"
