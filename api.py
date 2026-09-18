"""FastAPI backend wiring HTTP requests to the response pipeline."""

import logging
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memory import store
from respond import get_response

logger = logging.getLogger(__name__)

app = FastAPI(title="Darukaa.Earth Biodiversity Chatbot")


# --- Request/response models ---
# Pydantic validates untrusted input coming over the network.


class ChatRequest(BaseModel):
    """Incoming chat request with structured metrics and optional message."""

    session_id: str | None = None
    metrics: dict[str, float | str] = Field(default_factory=dict)
    message: str | None = None


class ChatResponse(BaseModel):
    """Validated recommendation response returned to the client."""

    session_id: str
    recommendations: list[dict]
    dropped_count: int
    needs_more_info: bool = False
    merged_metrics: dict = Field(default_factory=dict)
    retrieval_debug: dict = Field(default_factory=dict)
    message: str | None = None


# --- Endpoints ---


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Accept structured metrics for a session and return validated recommendations.

    If no session_id is provided, a new one is generated and returned. Free-text
    messages are parsed deterministically into metrics when reliable, and are
    also stored for conversational context.
    """
    session_id = request.session_id or str(uuid.uuid4())

    if request.message:
        store.add_message(session_id, "user", request.message)

    try:
        result = get_response(session_id, request.metrics, request.message or "")
    except ValueError as error:
        logger.warning("Bad chat request for session %s: %s", session_id, error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Unexpected chat failure for session %s", session_id)
        raise HTTPException(
            status_code=500,
            detail="The chatbot pipeline failed while generating a response.",
        ) from error

    return ChatResponse(
        session_id=session_id,
        recommendations=result.get("recommendations", []),
        dropped_count=int(result.get("dropped_count", 0)),
        needs_more_info=bool(result.get("needs_more_info", False)),
        merged_metrics=result.get("merged_metrics", {}),
        retrieval_debug=result.get("retrieval_debug", {}),
        message=result.get("message"),
    )


@app.get("/session/{session_id}")
def get_session(session_id: str) -> dict:
    """
    Return metrics and recent history for a session.

    This endpoint creates an empty session if the ID does not exist, which keeps
    demo/debug behavior simple and predictable.
    """
    session = store.get_or_create(session_id)
    return {
        "session_id": session.session_id,
        "metrics": session.metrics,
        "history": store.get_recent_history(session_id),
    }


@app.get("/health")
def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}
