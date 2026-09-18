"""Build the situation summary, assemble the prompt, and call the LLM."""

import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

_client: Any | None = None

DESCRIPTOR_SUMMARY_PHRASINGS = {
    "fragmentation": "habitat fragmentation",
    "connectivity": "connectivity and corridors",
    "degradation": "degradation",
    "water_limited": "water-limited conditions",
    "erosion_risk": "erosion risk",
    "slope": "sloped terrain",
    "waterways": "waterways and riparian context",
    "perennial_cover": "perennial cover",
    "salinity": "salinity",
}


# --- 1. Situation summary ---


def build_situation_summary(
    classifications: dict,
    interactions: list,
    descriptors: set[str] | None = None,
) -> str:
    """
    Turn classified metrics and detected interaction rules into a short summary.

    This string is used both as the retrieval query and as context inside the
    LLM prompt.
    """
    metric_parts: list[str] = []
    for metric_name in sorted(classifications):
        result = classifications[metric_name]
        value = result.get("value")
        band_label = result.get("band_label")
        if value == band_label:
            metric_parts.append(f"{metric_name}: {band_label}.")
        else:
            metric_parts.append(f"{metric_name}: {band_label} ({value}).")

    if interactions:
        interaction_text = "; ".join(_describe_interaction(rule) for rule in interactions)
        metric_parts.append(f"Detected interactions: {interaction_text}.")
    else:
        metric_parts.append("Detected interactions: none.")

    if descriptors:
        phrasings = [
            DESCRIPTOR_SUMMARY_PHRASINGS.get(descriptor, descriptor)
            for descriptor in sorted(descriptors)
        ]
        metric_parts.append("User context descriptors: " + ", ".join(phrasings) + ".")

    return " ".join(metric_parts)


def _describe_interaction(rule: object) -> str:
    """Return a readable rule phrase from a Rule-like object."""
    description = getattr(rule, "description", "")
    rule_id = getattr(rule, "rule_id", "unknown_interaction")
    return description.rstrip(".") if description else str(rule_id).replace("_", " ")


# --- 2. Prompt assembly ---


OUTPUT_SCHEMA_INSTRUCTIONS = """
Respond ONLY with a JSON object in this exact shape, nothing else:
{
  "recommendations": [
    {
      "recommendation": "string - the specific action to take",
      "mechanism": "string - why it works, scientifically",
      "metrics_impacted": ["string", ...],
      "time_horizon": "short" | "medium" | "long",
      "source": "string - must exactly match one of the provided source names"
    }
  ]
}
Provide 1-3 recommendations. Only include a recommendation if it is
genuinely supported by the retrieved context below. If the retrieved
context does not support any strong recommendation, return fewer
recommendations, even zero, rather than inventing one. Use ONLY
information from the retrieved context -- do not use outside knowledge,
even if you believe it to be true.

Quantitative estimates: When the retrieved evidence contains a quantitative
estimate relevant to a recommendation -- a percentage, a rate per hectare, a
timespan, a numeric range, or a ratio -- prioritize including that estimate in
the recommendation text over a purely qualitative description. Use the figure
exactly as it appears in the evidence. Do not invent, round, extrapolate, or
combine numbers from different evidence passages. If the retrieved evidence does
not contain a relevant number, do not add one.

Specific practices and species: When the retrieved evidence names a specific
plant species (for example chickpea, lentil, hairy vetch, clover, cowpea), a
plant functional group (for example legume, grass, deep-rooted crop, forb), or
a specific practice (for example three-crop rotation, mixed cover crop,
contour-planted tree rows), name it explicitly in the recommendation text
rather than describing the practice generically. Only use names that appear in
the retrieved evidence.
""".strip()


def build_prompt(situation_summary: str, retrieved_chunks: list, history: list) -> str:
    """
    Assemble the full prompt with situation, evidence, history, and output schema.

    Retrieved chunks are clearly delimited with exact source names so the model
    can cite sources that the validator can verify by exact string matching.
    """
    context_blocks = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[CONTEXT CHUNK {index}]",
                    f"SOURCE: {chunk['source']}",
                    f"INTERVENTION: {chunk['intervention']}",
                    f"METRIC: {chunk['metric']}",
                    f"REGION: {chunk['region']}",
                    f"SIMILARITY_SCORE: {chunk['similarity_score']}",
                    "TEXT:",
                    str(chunk["text"]),
                    f"[/CONTEXT CHUNK {index}]",
                ],
            ),
        )

    history_lines = []
    for message in history:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        history_lines.append(f"{role}: {content}")

    return "\n\n".join(
        [
            "You are Darukaa.Earth's biodiversity intelligence assistant.",
            "Use only the retrieved context. Do not invent sources, figures, or mechanisms.",
            f"SITUATION SUMMARY:\n{situation_summary}",
            "RETRIEVED CONTEXT:\n" + "\n\n".join(context_blocks),
            "RECENT CONVERSATION HISTORY:\n" + ("\n".join(history_lines) or "None."),
            OUTPUT_SCHEMA_INSTRUCTIONS,
        ],
    )


# --- 3. LLM call ---


def generate_recommendations(prompt: str) -> dict:
    """
    Call Gemini and parse its response as the expected recommendation JSON.

    Raises ValueError if GEMINI_API_KEY is empty, the model returns invalid JSON,
    or the parsed JSON does not match the expected high-level shape.
    """
    client = _get_client()
    response = client.models.generate_content(
        model=LLM_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=LLM_TEMPERATURE),
    )
    response_text = _strip_json_fences(str(response.text))

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError("Gemini response was not valid JSON") from error

    _validate_output_shape(parsed)
    return parsed


def _get_client() -> Any:
    """Create and cache the Gemini client once per process."""
    global _client
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is empty; set it before calling Gemini.")
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _strip_json_fences(text: str) -> str:
    """Remove markdown JSON fences if the model includes them."""
    stripped = text.strip()
    fenced_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return fenced_match.group(1).strip() if fenced_match else stripped


def _validate_output_shape(parsed: object) -> None:
    """Validate the expected top-level recommendation payload shape."""
    if not isinstance(parsed, dict):
        raise ValueError("Gemini JSON must be an object.")

    recommendations = parsed.get("recommendations")
    if not isinstance(recommendations, list):
        raise ValueError('Gemini JSON must contain a "recommendations" list.')

    required_fields = {"recommendation", "mechanism", "metrics_impacted", "time_horizon", "source"}
    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            raise ValueError("Each recommendation must be an object.")
        missing = required_fields - recommendation.keys()
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Recommendation missing required field(s): {missing_text}")
        if recommendation["time_horizon"] not in {"short", "medium", "long"}:
            raise ValueError("Recommendation time_horizon must be short, medium, or long.")
        if not isinstance(recommendation["metrics_impacted"], list):
            raise ValueError("Recommendation metrics_impacted must be a list.")
