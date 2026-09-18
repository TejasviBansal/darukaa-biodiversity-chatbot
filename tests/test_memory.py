"""Tests for the in-memory session store."""

import memory
from memory import SessionStore


def test_create_session_and_set_metrics() -> None:
    """A created session receives metrics through the store."""
    session_store = SessionStore()
    session = session_store.create()

    session_store.set_metric(session.session_id, "soil_ph", 7.1)

    assert session_store.get_or_create(session.session_id).metrics == {"soil_ph": 7.1}


def test_history_trims_to_configured_limit() -> None:
    """History keeps only the configured number of recent messages."""
    session_store = SessionStore()
    session_id = "history-session"

    for index in range(memory.SESSION_HISTORY_MAX_TURNS + 3):
        session_store.add_message(session_id, "user", f"message-{index}")

    history = session_store.get_recent_history(session_id)

    assert len(history) == memory.SESSION_HISTORY_MAX_TURNS
    assert history[0]["content"] == "message-3"
    assert history[-1]["content"] == f"message-{memory.SESSION_HISTORY_MAX_TURNS + 2}"


def test_sessions_do_not_leak_into_each_other() -> None:
    """Metrics and history stay isolated between session IDs."""
    session_store = SessionStore()

    session_store.set_metric("one", "soil_ph", 6.8)
    session_store.set_metric("two", "soil_ph", 8.2)
    session_store.add_message("one", "user", "hello from one")
    session_store.add_message("two", "user", "hello from two")

    assert session_store.get_or_create("one").metrics == {"soil_ph": 6.8}
    assert session_store.get_or_create("two").metrics == {"soil_ph": 8.2}
    assert session_store.get_recent_history("one") == [
        {"role": "user", "content": "hello from one"},
    ]
    assert session_store.get_recent_history("two") == [
        {"role": "user", "content": "hello from two"},
    ]
