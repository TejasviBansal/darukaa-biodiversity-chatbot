"""Tests for the FastAPI backend."""

from fastapi.testclient import TestClient

import api
from api import app
from memory import store

client = TestClient(app)


def test_health_returns_ok() -> None:
    """The health endpoint returns a simple liveness payload."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_generates_session_id_when_missing(monkeypatch) -> None:
    """A chat request without session_id receives a generated ID."""
    monkeypatch.setattr(
        api,
        "get_response",
        lambda session_id, metrics, message="": {
            "recommendations": [
                {
                    "recommendation": "Use cover crops.",
                    "mechanism": "They protect soil.",
                    "metrics_impacted": ["erosion_risk"],
                    "time_horizon": "medium",
                    "source": "Test Source",
                    "confidence": "medium",
                },
            ],
            "dropped_count": 0,
            "needs_more_info": False,
        },
    )

    response = client.post(
        "/chat",
        json={"metrics": {"soil_ph": 7.0}},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["session_id"]
    assert payload["needs_more_info"] is False
    assert payload["recommendations"][0]["recommendation"] == "Use cover crops."


def test_chat_returns_400_for_invalid_metric_value(monkeypatch) -> None:
    """ValueError from the pipeline is treated as bad input."""

    def raise_bad_input(session_id, metrics, message=""):
        raise ValueError("Value for soil_ph must be between 0.0 and 14.0: 999")

    monkeypatch.setattr(api, "get_response", raise_bad_input)

    response = client.post("/chat", json={"metrics": {"soil_ph": 999}})

    assert response.status_code == 400
    assert "soil_ph" in response.json()["detail"]


def test_chat_returns_500_for_unexpected_exception(monkeypatch) -> None:
    """Unexpected errors become clean 500 responses without stack traces."""

    def raise_unexpected(session_id, metrics, message=""):
        raise RuntimeError("database exploded")

    monkeypatch.setattr(api, "get_response", raise_unexpected)

    response = client.post("/chat", json={"metrics": {"soil_ph": 7.0}})

    assert response.status_code == 500
    assert response.json() == {
        "detail": "The chatbot pipeline failed while generating a response.",
    }


def test_get_session_returns_existing_session_shape() -> None:
    """The session endpoint returns metrics and recent history."""
    store.reset()
    store.set_metric("session-1", "soil_ph", 6.8)
    store.add_message("session-1", "user", "hello")

    response = client.get("/session/session-1")
    payload = response.json()

    assert response.status_code == 200
    assert payload == {
        "session_id": "session-1",
        "metrics": {"soil_ph": 6.8},
        "history": [{"role": "user", "content": "hello"}],
    }
