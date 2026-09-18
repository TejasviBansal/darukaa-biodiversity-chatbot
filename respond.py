"""Run the full classify, retrieve, generate, and validate response pipeline."""

import logging

from classifier import classify_all
from generator import build_prompt, build_situation_summary, generate_recommendations
from interactions import detect_interactions
from knowledge import retrieve
from memory import store
from validator import validate_recommendations

logger = logging.getLogger(__name__)


def get_response(session_id: str, metrics: dict) -> dict:
    """
    Run one chatbot turn from raw metrics to validated recommendations.

    If retrieval returns no evidence chunks, skip the LLM call and return a
    fallback instead of generating from an empty context.
    """
    classifications = classify_all(metrics)
    interaction_rules = detect_interactions(classifications)
    situation_summary = build_situation_summary(classifications, interaction_rules)
    retrieved_chunks = retrieve(situation_summary)

    if not retrieved_chunks:
        fallback_message = (
            "I need more site detail or stronger matching evidence before recommending actions."
        )
        fallback = {
            "recommendations": [],
            "dropped_count": 0,
            "message": fallback_message,
        }
        store.add_message(session_id, "assistant", fallback_message)
        return fallback

    history = store.get_recent_history(session_id)
    prompt = build_prompt(situation_summary, retrieved_chunks, history)
    llm_output = generate_recommendations(prompt)
    validated = validate_recommendations(llm_output, retrieved_chunks)
    store.add_message(session_id, "assistant", _summarize_result(validated))
    return validated


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
