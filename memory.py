"""In-memory conversation session store. Not persistent — resets on restart, by design."""

import logging
import threading
from dataclasses import dataclass, field
from uuid import uuid4

from config import SESSION_HISTORY_MAX_TURNS

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """A single chat session with collected metrics and recent message history."""

    session_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)


class SessionStore:
    """Thread-safe in-memory registry of active sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        """Create a new session with a generated UUID."""
        with self._lock:
            session = Session(session_id=str(uuid4()))
            self._sessions[session.session_id] = session
            return session

    def get_or_create(self, session_id: str) -> Session:
        """Fetch an existing session or create one with the given ID."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id=session_id)
            return self._sessions[session_id]

    def set_metric(self, session_id: str, name: str, value: float) -> None:
        """Set or overwrite one metric value for a session."""
        with self._lock:
            session = self._sessions.setdefault(session_id, Session(session_id=session_id))
            session.metrics[name] = value

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message, trimming history to SESSION_HISTORY_MAX_TURNS."""
        with self._lock:
            session = self._sessions.setdefault(session_id, Session(session_id=session_id))
            session.history.append({"role": role, "content": content})
            session.history = session.history[-SESSION_HISTORY_MAX_TURNS:]

    def get_recent_history(self, session_id: str) -> list[dict[str, str]]:
        """Return the most recent messages, up to the configured limit."""
        with self._lock:
            session = self._sessions.setdefault(session_id, Session(session_id=session_id))
            return list(session.history[-SESSION_HISTORY_MAX_TURNS:])

    def reset(self) -> None:
        """Clear all sessions — used by tests."""
        with self._lock:
            self._sessions.clear()


store = SessionStore()
