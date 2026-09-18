"""Tests for response-pipeline helpers."""

import respond
from config import METRIC_LABELS
from respond import _summarize_result


def test_get_response_asks_for_all_core_metrics_when_none_provided(monkeypatch) -> None:
    """Empty input should ask for the three core metrics before retrieval."""

    def fail_retrieve(query):
        raise AssertionError("retrieve should not be called")

    monkeypatch.setattr(respond, "retrieve", fail_retrieve)
    respond.store.reset()

    result = respond.get_response("session-empty", {}, "")

    assert result["needs_more_info"] is True
    assert result["recommendations"] == []
    assert METRIC_LABELS["soil_organic_carbon_pct"] in result["message"]
    assert METRIC_LABELS["rainfall_mm_year"] in result["message"]
    assert METRIC_LABELS["land_use_category"] in result["message"]


def test_get_response_acknowledges_biodiversity_and_missing_core_metrics(monkeypatch) -> None:
    """One core metric plus a biodiversity mention should ask only for missing core details."""

    def fail_retrieve(query):
        raise AssertionError("retrieve should not be called")

    monkeypatch.setattr(respond, "retrieve", fail_retrieve)
    respond.store.reset()

    result = respond.get_response(
        "session-one-core",
        {"rainfall_mm_year": 350.0},
        "biodiversity is declining",
    )

    assert result["needs_more_info"] is True
    assert "You mentioned biodiversity" in result["message"]
    assert METRIC_LABELS["soil_organic_carbon_pct"] in result["message"]
    assert METRIC_LABELS["land_use_category"] in result["message"]
    assert METRIC_LABELS["rainfall_mm_year"] not in result["message"]


def test_get_response_proceeds_when_two_core_metrics_are_provided(monkeypatch) -> None:
    """Two core metrics are enough to continue into retrieval and generation."""
    retrieve_called = False

    def fake_retrieve(query):
        nonlocal retrieve_called
        retrieve_called = True
        return [{"source": "Test Source", "text": "Test evidence.", "similarity_score": 0.5}]

    monkeypatch.setattr(respond, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        respond,
        "generate_recommendations",
        lambda prompt: {"recommendations": []},
    )
    monkeypatch.setattr(
        respond,
        "validate_recommendations",
        lambda llm_output, chunks: {"recommendations": [], "dropped_count": 0},
    )
    monkeypatch.setattr(respond, "build_prompt", lambda summary, chunks, history: "prompt")
    respond.store.reset()

    result = respond.get_response(
        "session-two-core",
        {"soil_organic_carbon_pct": 0.3, "rainfall_mm_year": 350.0},
    )

    assert retrieve_called is True
    assert result["needs_more_info"] is False


def test_get_response_proceeds_with_free_text_extracted_metrics(monkeypatch) -> None:
    """Free-text extracted rainfall and land use should satisfy the core-metric check."""
    captured_metrics: dict = {}

    def fake_classify_all(metrics):
        captured_metrics.update(metrics)
        return {
            metric_name: {
                "metric_name": metric_name,
                "value": value,
                "band_label": value,
                "source": "test",
            }
            for metric_name, value in metrics.items()
        }

    monkeypatch.setattr(respond, "classify_all", fake_classify_all)
    monkeypatch.setattr(respond, "detect_interactions", lambda classifications: [])
    monkeypatch.setattr(
        respond,
        "build_situation_summary",
        lambda classifications, rules, descriptors=None: "summary",
    )
    monkeypatch.setattr(
        respond,
        "retrieve",
        lambda query: [
            {"source": "Test Source", "text": "Test evidence.", "similarity_score": 0.5},
        ],
    )
    monkeypatch.setattr(respond, "build_prompt", lambda summary, chunks, history: "prompt")
    monkeypatch.setattr(
        respond,
        "generate_recommendations",
        lambda prompt: {"recommendations": []},
    )
    monkeypatch.setattr(
        respond,
        "validate_recommendations",
        lambda llm_output, chunks: {"recommendations": [], "dropped_count": 0},
    )
    respond.store.reset()

    result = respond.get_response(
        "session-free-text",
        {},
        "biodiversity is declining, I grow only wheat in a semi-arid area with 350mm rainfall",
    )

    assert captured_metrics["species_richness_trend"] == "declining"
    assert captured_metrics["land_use_category"] == "monoculture"
    assert captured_metrics["rainfall_mm_year"] == 350.0
    assert result["needs_more_info"] is False


def test_structured_values_win_over_free_text_extraction(monkeypatch) -> None:
    """Structured metric values are not overwritten by free-text extraction."""
    captured_metrics: dict = {}

    def fake_classify_all(metrics):
        captured_metrics.update(metrics)
        return {}

    monkeypatch.setattr(respond, "classify_all", fake_classify_all)
    respond.store.reset()

    respond.get_response("session-structured-wins", {"soil_ph": 7.0}, "soil pH is 5.5")

    assert captured_metrics["soil_ph"] == 7.0


def test_single_non_core_metric_triggers_clarification(monkeypatch) -> None:
    """One non-core metric is not enough to run retrieval and generation."""

    def fail_retrieve(query):
        raise AssertionError("retrieve should not be called")

    monkeypatch.setattr(respond, "retrieve", fail_retrieve)
    respond.store.reset()

    result = respond.get_response("session-single-non-core", {"soil_ph": 7.0}, "")

    assert result["needs_more_info"] is True


def test_clarifying_question_does_not_contradict_missing_topics(monkeypatch) -> None:
    """Don't say 'you mentioned soil' when soil is exactly what's missing."""

    def fail_retrieve(query):
        raise AssertionError("retrieve should not be called")

    monkeypatch.setattr(respond, "retrieve", fail_retrieve)
    respond.store.reset()

    result = respond.get_response("s1", {}, "the soil is degraded and I grow monoculture")

    assert result["needs_more_info"] is True
    assert "you mentioned soil" not in result["message"].lower()
    assert "You mentioned land use" in result["message"]


def test_get_response_returns_fallback_when_llm_generation_fails(monkeypatch) -> None:
    """Temporary Gemini failures should not break the chat turn."""

    monkeypatch.setattr(
        respond,
        "classify_all",
        lambda metrics: {
            "soil_ph": {
                "metric_name": "soil_ph",
                "value": 7.0,
                "band_label": "neutral",
                "source": "test",
            },
        },
    )
    monkeypatch.setattr(respond, "detect_interactions", lambda classifications: [])
    monkeypatch.setattr(
        respond,
        "build_situation_summary",
        lambda classifications, rules, descriptors=None: "soil_ph",
    )
    monkeypatch.setattr(
        respond,
        "retrieve",
        lambda query: [
            {
                "source": "Test Source",
                "text": "Test evidence.",
                "similarity_score": 0.5,
            },
        ],
    )
    monkeypatch.setattr(respond, "build_prompt", lambda summary, chunks, history: "prompt")

    def raise_overloaded(prompt):
        raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr(respond, "generate_recommendations", raise_overloaded)
    respond.store.reset()

    result = respond.get_response(
        "session-llm-overload",
        {"soil_organic_carbon_pct": 0.3, "rainfall_mm_year": 350.0},
    )

    assert result["recommendations"] == []
    assert result["dropped_count"] == 0
    assert result["needs_more_info"] is False
    assert result["message"] == (
        "The recommendation model is temporarily unavailable. "
        "Please try again in a few minutes."
    )
    assert result["merged_metrics"] == {
        "soil_organic_carbon_pct": 0.3,
        "rainfall_mm_year": 350.0,
    }
    assert result["retrieval_debug"]["situation_summary"] == "soil_ph"
    assert result["retrieval_debug"]["chunks"]
    assert respond.store.get_recent_history("session-llm-overload") == [
        {
            "role": "assistant",
            "content": "The recommendation model is temporarily unavailable. "
            "Please try again in a few minutes.",
        },
    ]


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


def test_clarifying_question_acknowledges_descriptors(monkeypatch) -> None:
    """The clarifying question should acknowledge fragmentation and connectivity."""

    def fail_retrieve(query):
        raise AssertionError("retrieve should not be called")

    monkeypatch.setattr(respond, "retrieve", fail_retrieve)
    respond.store.reset()

    result = respond.get_response(
        "session-descriptors",
        {},
        "Our tea plantation borders a tropical reserve, but habitat fragmentation "
        "is dropping native pollinator and avian diversity; what land-use "
        "interventions will reconnect these corridors?",
    )

    assert result["needs_more_info"] is True
    message_lower = result["message"].lower()
    assert "biodiversity" in message_lower
    assert "habitat fragmentation" in message_lower
    assert "reconnecting habitat corridors" in message_lower
    assert METRIC_LABELS["soil_organic_carbon_pct"] in result["message"]
    assert METRIC_LABELS["rainfall_mm_year"] in result["message"]
    assert METRIC_LABELS["land_use_category"] in result["message"]


def test_clarifying_question_suppresses_water_limited_when_rainfall_missing(
    monkeypatch,
) -> None:
    """Semi-arid phrasing should not be acknowledged while asking for rainfall."""

    def fail_retrieve(query):
        raise AssertionError("retrieve should not be called")

    monkeypatch.setattr(respond, "retrieve", fail_retrieve)
    respond.store.reset()

    result = respond.get_response(
        "session-water-limited",
        {"land_use_category": "monoculture"},
        "dryland wheat system",
    )

    assert result["needs_more_info"] is True
    assert "water-limited conditions" not in result["message"]
    assert METRIC_LABELS["rainfall_mm_year"] in result["message"]


def test_clarifying_question_acknowledges_extracted_non_core_metric(monkeypatch) -> None:
    """A provided non-core metric should be acknowledged, not ignored."""

    def fail_retrieve(query):
        raise AssertionError("retrieve should not be called")

    monkeypatch.setattr(respond, "retrieve", fail_retrieve)
    respond.store.reset()

    result = respond.get_response(
        "session-ph-ack",
        {"soil_ph": 8.2},
        "We have 5 hectares of degraded peri-urban scrubland with compacted clay "
        "and alkaline pH (8.2); how can we accelerate soil carbon and native plant regeneration?",
    )

    assert result["needs_more_info"] is True
    message_lower = result["message"].lower()
    assert "soil ph" in message_lower
    assert "8.2" in result["message"]
    assert "land degradation" in message_lower
    assert METRIC_LABELS["soil_organic_carbon_pct"] in result["message"]
    assert METRIC_LABELS["rainfall_mm_year"] in result["message"]
    assert METRIC_LABELS["land_use_category"] in result["message"]


def test_clarifying_question_handles_erosion_erosion_context(monkeypatch) -> None:
    """Erosion/waterways/slope context should be acknowledged when metrics are missing."""

    def fail_retrieve(query):
        raise AssertionError("retrieve should not be called")

    monkeypatch.setattr(respond, "retrieve", fail_retrieve)
    respond.store.reset()

    result = respond.get_response(
        "session-erosion",
        {},
        "Heavy seasonal rainfall on newly deforested, sloped red-loam soils is "
        "causing massive topsoil erosion and downstream siltation; what deep-rooting "
        "species and ground cover combinations prevent this degradation?",
    )

    assert result["needs_more_info"] is True
    message_lower = result["message"].lower()
    assert "erosion risk" in message_lower
    assert "sloped terrain" in message_lower
    assert "nearby waterways" in message_lower


def test_situation_summary_includes_descriptors_for_retrieval(monkeypatch) -> None:
    """Descriptors from the message must flow into the situation summary used for retrieval."""
    captured_summary: dict = {}

    def fake_retrieve(query):
        captured_summary["query"] = query
        return [{"source": "Test Source", "text": "Test evidence.", "similarity_score": 0.5}]

    monkeypatch.setattr(respond, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        respond,
        "generate_recommendations",
        lambda prompt: {"recommendations": []},
    )
    monkeypatch.setattr(
        respond,
        "validate_recommendations",
        lambda llm_output, chunks: {"recommendations": [], "dropped_count": 0},
    )
    monkeypatch.setattr(respond, "build_prompt", lambda summary, chunks, history: "prompt")
    respond.store.reset()

    respond.get_response(
        "session-descriptor-retrieval",
        {"land_use_category": "monoculture", "soil_organic_carbon_pct": 0.3},
        "habitat fragmentation is dropping pollinator diversity; "
        "how do I reconnect these corridors?",
    )

    query = captured_summary["query"]
    assert "User context descriptors:" in query
    assert "habitat fragmentation" in query
    assert "connectivity and corridors" in query
