"""Validate LLM-generated recommendations against retrieved evidence."""

import logging
import re

logger = logging.getLogger(__name__)


# --- 1. Source verification ---


def verify_source(recommendation: dict, retrieved_chunks: list) -> bool:
    """
    Return True if recommendation["source"] exactly matches a retrieved source.
    """
    source = recommendation.get("source")
    return any(source == chunk.get("source") for chunk in retrieved_chunks)


# --- 2. Numeric/entity grounding check ---


def _extract_numbers(text: str) -> list[str]:
    """
    Extract numeric tokens from text.

    Handles percentages, decimals, ranges like "15-25", and compact units such
    as "0.5 t/ha/year" by checking the numeric portion as an exact substring.
    """
    return re.findall(r"\b\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?%?", text)


def check_numbers_grounded(recommendation: dict, retrieved_chunks: list) -> bool:
    """
    Return True only when every number in the recommendation appears in evidence.

    Qualitative recommendations with no numeric claims pass this check.
    """
    text_to_check = " ".join(
        [
            str(recommendation.get("recommendation", "")),
            str(recommendation.get("mechanism", "")),
        ],
    )
    numbers = _extract_numbers(text_to_check)
    if not numbers:
        return True

    evidence_text = "\n".join(str(chunk.get("text", "")) for chunk in retrieved_chunks)
    return all(number in evidence_text for number in numbers)


# --- 3. Confidence scoring from real similarity, not LLM self-assessment ---


def compute_confidence(recommendation: dict, retrieved_chunks: list) -> str:
    """
    Map the best source-matched retrieval similarity to a confidence tier.
    """
    source = recommendation.get("source")
    matching_scores = [
        float(chunk.get("similarity_score", 0.0))
        for chunk in retrieved_chunks
        if chunk.get("source") == source
    ]
    if not matching_scores:
        return "low"

    best_score = max(matching_scores)
    if best_score >= 0.55:
        return "high"
    if best_score >= 0.45:
        return "medium"
    return "low"


# --- 4. Full validation pass ---


def validate_recommendations(llm_output: dict, retrieved_chunks: list) -> dict:
    """
    Drop ungrounded recommendations and attach confidence to surviving ones.
    """
    validated: list[dict] = []
    dropped_count = 0

    for recommendation in llm_output.get("recommendations", []):
        if not verify_source(recommendation, retrieved_chunks):
            logger.warning(
                "Dropping recommendation with unverified source: %s",
                recommendation.get("source"),
            )
            dropped_count += 1
            continue

        if not check_numbers_grounded(recommendation, retrieved_chunks):
            logger.warning(
                "Dropping recommendation with ungrounded numeric claim: %s",
                recommendation.get("recommendation"),
            )
            dropped_count += 1
            continue

        enriched = dict(recommendation)
        enriched["confidence"] = compute_confidence(recommendation, retrieved_chunks)
        validated.append(enriched)

    return {"recommendations": validated, "dropped_count": dropped_count}
