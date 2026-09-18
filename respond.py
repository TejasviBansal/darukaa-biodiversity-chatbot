"""Run the full classify, retrieve, generate, and validate response pipeline."""

import logging

from classifier import classify_all
from config import CORE_METRICS, METRIC_LABELS, MIN_CORE_METRICS_REQUIRED
from extractor import detect_descriptors, detect_topics, extract_metrics
from generator import build_prompt, build_situation_summary, generate_recommendations
from interactions import detect_interactions
from knowledge import retrieve
from memory import store
from validator import validate_recommendations

logger = logging.getLogger(__name__)

TOPIC_TO_CORE_METRIC = {
    "soil": "soil_organic_carbon_pct",
    "water_climate": "rainfall_mm_year",
    "land_use": "land_use_category",
}

DESCRIPTOR_PHRASINGS = {
    "fragmentation": "habitat fragmentation",
    "connectivity": "reconnecting habitat corridors",
    "degradation": "land degradation",
    "water_limited": "water-limited conditions",
    "erosion_risk": "erosion risk",
    "slope": "sloped terrain",
    "waterways": "nearby waterways",
    "perennial_cover": "perennial cover",
    "salinity": "high soil salinity",
}

DESCRIPTOR_SUPPRESSED_BY_MISSING_METRIC = {
    "water_limited": "rainfall_mm_year",
    "degradation": "soil_organic_carbon_pct",
}

NON_CORE_METRIC_PHRASINGS = {
    "soil_ph": "soil pH",
    "temperature_deviation_c": "temperature deviation",
    "ground_cover_pct": "ground cover",
    "species_richness_trend": "species richness trend",
}


def get_response(session_id: str, metrics: dict, message: str = "") -> dict:
    """
    Run one chatbot turn from raw metrics to validated recommendations.

    If retrieval returns no evidence chunks, skip the LLM call and return a
    fallback instead of generating from an empty context.
    """
    merged = dict(metrics)
    extracted = extract_metrics(message)
    for key, value in extracted.items():
        merged.setdefault(key, value)

    topics = detect_topics(message)
    descriptors = detect_descriptors(message)
    classifications = classify_all(merged)

    provided_core = [metric for metric in CORE_METRICS if metric in merged]
    if len(provided_core) < MIN_CORE_METRICS_REQUIRED:
        clarification = _build_clarifying_question(merged, topics, descriptors)
        store.add_message(session_id, "assistant", clarification)
        return {
            "recommendations": [],
            "dropped_count": 0,
            "needs_more_info": True,
            "message": clarification,
            "merged_metrics": merged,
            "retrieval_debug": _retrieval_debug(None, []),
        }

    interaction_rules = detect_interactions(classifications)
    situation_summary = build_situation_summary(classifications, interaction_rules, descriptors)
    retrieved_chunks = retrieve(situation_summary)

    if not retrieved_chunks:
        fallback_message = (
            "I couldn't find matching evidence for this combination of conditions "
            "in the knowledge base."
        )
        fallback = {
            "recommendations": [],
            "dropped_count": 0,
            "needs_more_info": False,
            "message": fallback_message,
            "merged_metrics": merged,
            "retrieval_debug": _retrieval_debug(situation_summary, []),
        }
        store.add_message(session_id, "assistant", fallback_message)
        return fallback

    history = store.get_recent_history(session_id)
    prompt = build_prompt(situation_summary, retrieved_chunks, history)
    try:
        llm_output = generate_recommendations(prompt)
    except Exception as error:
        logger.warning("LLM generation failed for session %s: %s", session_id, error)
        fallback_message = (
            "The recommendation model is temporarily unavailable. "
            "Please try again in a few minutes."
        )
        fallback = {
            "recommendations": [],
            "dropped_count": 0,
            "needs_more_info": False,
            "message": fallback_message,
            "merged_metrics": merged,
            "retrieval_debug": _retrieval_debug(situation_summary, retrieved_chunks),
        }
        store.add_message(session_id, "assistant", fallback_message)
        return fallback

    validated = validate_recommendations(llm_output, retrieved_chunks)
    validated["needs_more_info"] = False
    validated["merged_metrics"] = merged
    validated["retrieval_debug"] = _retrieval_debug(situation_summary, retrieved_chunks)
    store.add_message(session_id, "assistant", _summarize_result(validated))
    return validated


def _retrieval_debug(situation_summary: str | None, chunks: list[dict]) -> dict:
    """Return compact retrieval details for UI display."""
    return {
        "situation_summary": situation_summary,
        "chunks": [
            {
                "source": chunk.get("source", ""),
                "similarity_score": chunk.get("similarity_score", 0.0),
                "text": str(chunk.get("text", ""))[:300],
            }
            for chunk in chunks
        ],
    }


def _build_clarifying_question(
    metrics: dict,
    topics: set[str],
    descriptors: set[str],
) -> str:
    """Build a short question naming missing core metrics and acknowledging context."""
    missing = [metric for metric in CORE_METRICS if metric not in metrics]
    missing_labels = ", ".join(METRIC_LABELS[metric] for metric in missing)

    acknowledgements: list[str] = []
    if "biodiversity" in topics:
        acknowledgements.append("you mentioned biodiversity")
    for topic, core_metric in TOPIC_TO_CORE_METRIC.items():
        if topic in topics and core_metric not in missing:
            acknowledgements.append(f"you mentioned {_topic_phrase(topic)}")

    descriptor_phrases: list[str] = []
    for descriptor in sorted(descriptors):
        suppressed_by = DESCRIPTOR_SUPPRESSED_BY_MISSING_METRIC.get(descriptor)
        has_non_core_context = any(metric in metrics for metric in NON_CORE_METRIC_PHRASINGS)
        if suppressed_by and suppressed_by in missing and not has_non_core_context:
            continue
        phrasing = DESCRIPTOR_PHRASINGS.get(descriptor)
        if phrasing:
            descriptor_phrases.append(phrasing)

    non_core_notes: list[str] = []
    for metric_name, phrasing in NON_CORE_METRIC_PHRASINGS.items():
        if metric_name not in metrics:
            continue
        value = metrics[metric_name]
        if isinstance(value, int | float):
            non_core_notes.append(f"your {phrasing} of {value}")
        else:
            non_core_notes.append(f"your {phrasing} ({value})")

    prefix_parts: list[str] = []
    if acknowledgements:
        prefix_parts.append("; ".join(acknowledgements).capitalize())
    if descriptor_phrases:
        prefix_parts.append(
            "I understand you are dealing with " + ", ".join(descriptor_phrases),
        )
    if non_core_notes:
        prefix_parts.append("I noted " + ", ".join(non_core_notes))

    if prefix_parts:
        prefix = ". ".join(prefix_parts)
        return f"{prefix}. To recommend actions I also need: {missing_labels}."
    return f"To recommend actions, I need a few more site details: {missing_labels}."


def _topic_phrase(topic: str) -> str:
    """Return a readable phrase for a detected topic."""
    return {
        "soil": "soil",
        "water_climate": "rainfall/water",
        "land_use": "land use",
        "biodiversity": "biodiversity",
    }[topic]


def _summarize_result(result: dict) -> str:
    """Create a compact assistant-history summary from the validated result."""
    recommendations = result.get("recommendations", [])
    if not recommendations:
        return "No validated recommendations were returned."

    summary_lines = []
    for item in recommendations:
        recommendation = item.get("recommendation", "")
        time_horizon = item.get("time_horizon", "unknown")
        summary_lines.append(f"Recommended: {recommendation} (time horizon: {time_horizon})")
    return "\n".join(summary_lines)
