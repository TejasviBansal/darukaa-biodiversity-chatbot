"""Streamlit chat frontend for the Darukaa.Earth biodiversity chatbot."""

import os
import uuid
from typing import Any

import streamlit as st

from config import (
    CHROMA_PERSIST_DIR,
    CORE_METRICS,
    LAND_USE_CATEGORIES,
    METRIC_LABELS,
    SPECIES_RICHNESS_TRENDS,
)
from knowledge import build_knowledge_base as _ingest_knowledge_base
from knowledge import chunk_documents, load_corpus
from respond import get_response


def _ensure_knowledge_base() -> None:
    """
    Build the ChromaDB vector store on first run.

    Streamlit Community Cloud has an ephemeral filesystem, and chroma_store/ is
    not committed to git, so the ingestion must happen once per container start.
    Subsequent runs in the same container find the store and return immediately.
    """
    if os.path.exists(CHROMA_PERSIST_DIR):
        return
    with st.spinner("Building knowledge base (first run, ~30-60 seconds)..."):
        documents = load_corpus()
        chunks = chunk_documents(documents)
        _ingest_knowledge_base(chunks, reset=True)

st.set_page_config(
    page_title="Darukaa.Earth Biodiversity Chatbot",
    page_icon="🌱",
    initial_sidebar_state="expanded",
)
_ensure_knowledge_base()
st.title("🌱 Darukaa.Earth Biodiversity Intelligence Chatbot")


# --- Session state setup ---


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Structured input form ---


def _optional_number(label: str, help_text: str = "") -> float | None:
    """Render an optional numeric input that stays empty until filled."""
    return st.number_input(label, value=None, step=0.1, help=help_text)


def _select_optional(label: str, options: list[str]) -> str | None:
    """Render an optional selectbox with a no-value placeholder."""
    selected = st.selectbox(label, ["Not provided", *options])
    return None if selected == "Not provided" else selected


with st.sidebar:
    st.header("Site Metrics")
    with st.form("metrics_form"):
        soil_organic_carbon = _optional_number("Soil organic carbon (%)")
        rainfall = _optional_number("Rainfall (mm/year)")
        soil_ph = _optional_number("Soil pH")
        temperature_deviation = _optional_number("Temperature deviation (C)")
        ground_cover = _optional_number("Ground cover (%)")
        land_use = _select_optional("Land use category", LAND_USE_CATEGORIES)
        richness_trend = _select_optional("Species richness trend", SPECIES_RICHNESS_TRENDS)
        message = st.text_area("Describe your situation (or ask a question)", height=90)
        submitted = st.form_submit_button("Get Recommendations")


def _collect_metrics() -> dict[str, float | str]:
    """Collect only metrics the user actually filled in."""
    metrics: dict[str, float | str] = {}
    if soil_organic_carbon is not None:
        metrics["soil_organic_carbon_pct"] = soil_organic_carbon
    if rainfall is not None:
        metrics["rainfall_mm_year"] = rainfall
    if soil_ph is not None:
        metrics["soil_ph"] = soil_ph
    if temperature_deviation is not None:
        metrics["temperature_deviation_c"] = temperature_deviation
    if ground_cover is not None:
        metrics["ground_cover_pct"] = ground_cover
    if land_use is not None:
        metrics["land_use_category"] = land_use
    if richness_trend is not None:
        metrics["species_richness_trend"] = richness_trend
    return metrics


# --- Chat display ---


def _confidence_label(confidence: str) -> str:
    """Return a small colored confidence label for markdown rendering."""
    colors = {"high": "green", "medium": "orange", "low": "gray"}
    color = colors.get(confidence, "gray")
    return f":{color}[{confidence}]"


def _render_recommendation(item: dict[str, Any]) -> None:
    """Render one recommendation as a compact card-like block."""
    metrics = ", ".join(str(metric) for metric in item.get("metrics_impacted", [])) or "Not listed"
    confidence = _confidence_label(str(item.get("confidence", "low")))
    source = item.get("source") or "Not specified"
    with st.container(border=True):
        st.markdown(f"**Recommendation:** {item.get('recommendation', '')}")
        st.markdown(f"**Mechanism:** {item.get('mechanism', '')}")
        st.markdown(f"**Metrics impacted:** {metrics}")
        st.markdown(f"**Time horizon:** {item.get('time_horizon', 'unknown')}")
        st.markdown(f"**Source:** {source}")
        st.markdown(f"**Confidence:** {confidence}")


def _render_retrieval_details(content: dict[str, Any]) -> None:
    """Render compact retrieval debug details for the knowledge system."""
    with st.expander("Retrieval details (knowledge system)"):
        debug = content.get("retrieval_debug") or {}
        summary = debug.get("situation_summary")
        chunks = debug.get("chunks") or []

        if summary:
            st.markdown("**Situation summary sent to retrieval:**")
            st.code(summary, language=None)

        if not chunks:
            st.caption("No chunks were retrieved for this query.")
        else:
            st.markdown(f"**Retrieved chunks ({len(chunks)}):**")
            for chunk in chunks:
                source = chunk.get("source", "unknown source")
                score = chunk.get("similarity_score")
                score_text = f"{score:.3f}" if isinstance(score, int | float) else "n/a"
                st.markdown(f"- **{source}** - similarity: `{score_text}`")
                text = chunk.get("text", "")
                if text:
                    st.caption(text)


def _render_missing_core_caption(content: dict[str, Any]) -> None:
    """Suggest optional core metrics when recommendations already exist."""
    merged_metrics = content.get("merged_metrics") or {}
    missing_core = [metric for metric in CORE_METRICS if metric not in merged_metrics]
    if missing_core and content.get("recommendations"):
        labels = ", ".join(METRIC_LABELS[metric] for metric in missing_core)
        st.caption(f"For sharper recommendations, also provide: {labels}.")


def _render_assistant_content(content: dict[str, Any]) -> None:
    """Render an assistant response payload."""
    if content.get("message"):
        if content.get("needs_more_info"):
            st.warning(content["message"])
            return
        else:
            st.info(content["message"])
    recommendations = content.get("recommendations", [])
    if not recommendations:
        st.write("No validated recommendations were returned.")
    for item in recommendations:
        _render_recommendation(item)
    _render_retrieval_details(content)
    _render_missing_core_caption(content)
    dropped_count = content.get("dropped_count", 0)
    if dropped_count:
        st.caption(f"{dropped_count} ungrounded recommendation(s) were dropped.")


for message_item in st.session_state.messages:
    with st.chat_message(message_item["role"]):
        content = message_item["content"]
        if isinstance(content, dict):
            _render_assistant_content(content)
        else:
            st.write(content)


# --- Submit handling ---


if submitted:
    metrics_payload = _collect_metrics()
    note = message.strip() if message else None

    with st.spinner("Getting grounded recommendations..."):
        try:
            data = get_response(
                st.session_state.session_id,
                metrics_payload,
                note or "",
            )
        except ValueError as error:
            st.error(f"Invalid metric input: {error}")
            st.stop()
        except Exception as error:
            st.error(f"Pipeline error: {error}")
            st.stop()

        effective = data.get("merged_metrics") or {}
        if effective:
            metrics_line = "Effective metrics: " + ", ".join(
                f"{key}={value}" for key, value in effective.items()
            )
        else:
            metrics_line = "Effective metrics: none extracted"
        user_summary = f"{metrics_line}\n\nNote: {note}" if note else metrics_line
        st.session_state.messages.append({"role": "user", "content": user_summary})
        st.session_state.messages.append({"role": "assistant", "content": data})
        st.rerun()
