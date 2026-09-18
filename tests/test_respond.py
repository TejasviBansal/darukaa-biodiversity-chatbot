"""Tests for response-pipeline helpers."""

from respond import _summarize_result


def test_summarize_result_stores_recommendation_content_for_history() -> None:
    """Assistant history should contain readable recommendation text."""
    result = {
        "recommendations": [
            {
                "recommendation": "Plant a mixed cover crop after harvest.",
                "time_horizon": "medium",
            },
            {
                "recommendation": "Retain crop residues on bare soil.",
                "time_horizon": "short",
            },
        ],
        "dropped_count": 0,
    }

    summary = _summarize_result(result)

    assert "Recommended: Plant a mixed cover crop after harvest. (time horizon: medium)" in summary
    assert "Recommended: Retain crop residues on bare soil. (time horizon: short)" in summary
    assert "recommendation_count" not in summary


def test_summarize_result_handles_empty_recommendations() -> None:
    """Empty validated outputs still produce a useful history entry."""
    assert _summarize_result({"recommendations": []}) == (
        "No validated recommendations were returned."
    )
