"""Agent Session Manager — Phase 15: Session Context Handling and Persistence.

Provides comprehensive session management so the agent maintains proper context
across the entire conversation lifecycle, retains context from imported chats,
uses session history during execution, and supports auto-summarization when the
context window approaches its limit.

Named ``AgentSessionManager`` (not ``SessionManager``) to avoid collision with
the existing ``SessionManager`` in ``multi_terminal.py`` / ``session_manager.py``
which manages terminal sessions.

Architecture Note
-----------------
The assistant architecture remains stable.  The integrated model (any LLM
integrated through Ollama, API, etc.) operates dynamically through natural
language understanding and intent-driven execution.  This module is part of the
stable orchestration layer — it manages *session state*, not model behaviour.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import msvcrt  # Windows file locking
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False

try:
    import fcntl  # Unix file locking
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy / guarded imports — these may not be available in minimal environments
# ---------------------------------------------------------------------------
try:
    from proxima.agent.dynamic_tools.robust_nl_processor import (
        SessionContext,
        IntentType,
    )
    _SESSION_CONTEXT_AVAILABLE = True
except ImportError:
    _SESSION_CONTEXT_AVAILABLE = False

try:
    from proxima.intelligence.llm_router import LLMRouter, LLMRequest
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TodoItem:
    """A single todo / task entry tracked within a session.

    Attributes
    ----------
    content : str
        What needs to be done, written in imperative form (e.g. "Run tests").
    status : str
        One of ``"pending"``, ``"in_progress"``, or ``"completed"``.
    active_form : str
        Present-continuous form for UI display (e.g. "Running tests").
    """

    content: str
    status: str = "pending"          # pending | in_progress | completed
    active_form: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "status": self.status,
            "active_form": self.active_form,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TodoItem":
        return cls(
            content=data.get("content", ""),
            status=data.get("status", "pending"),
            active_form=data.get("active_form", ""),
        )


@dataclass
class SessionMessage:
    """A single message in the session history.

    Attributes
    ----------
    message_id : str
        Unique identifier for this message.
    role : str
        ``'user'``, ``'assistant'``, or ``'tool'``.
    content : str
        Text content of the message.
    timestamp : float
        Unix timestamp when the message was created.
    model : str or None
        Which model generated this (for assistant messages).
    tool_calls : list
        Tool calls made in this message (for assistant role).
    tool_results : list
        Tool results (for tool role messages).
    is_summary : bool
        ``True`` if this is an auto-generated summary message.
    metadata : dict
        Additional metadata (token count, etc.).
    """

    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    role: str = "user"
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    model: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    is_summary: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "model": self.model,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "is_summary": self.is_summary,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMessage":
        return cls(
            message_id=data.get("message_id", uuid.uuid4().hex[:16]),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            model=data.get("model"),
            tool_calls=data.get("tool_calls", []),
            tool_results=data.get("tool_results", []),
            is_summary=data.get("is_summary", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionState:
    """Complete state of an agent session.

    Attributes
    ----------
    session_id : str
        UUID for this session.
    title : str
        Auto-generated or user-set session title.
    created_at : float
        Unix timestamp of session creation.
    updated_at : float
        Unix timestamp of last activity.
    message_count : int
        Total messages (user + assistant + tool).
    prompt_tokens : int
        Cumulative prompt tokens used.
    completion_tokens : int
        Cumulative completion tokens used.
    cost : float
        Estimated cost (if using paid API).
    summary_message_id : str or None
        ID of the summary message (None if not yet summarized).
    todos : list of TodoItem
        Active todo list for this session.
    context : dict
        Serialized ``SessionContext`` dictionary.
    parent_session_id : str or None
        For sub-agent sessions — links to parent.
    is_sub_agent_session : bool
        True if this is a sub-agent task session.
    messages : list of SessionMessage
        Complete message history.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    title: str = "Untitled Session"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    message_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    summary_message_id: Optional[str] = None
    todos: List[TodoItem] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    parent_session_id: Optional[str] = None
    is_sub_agent_session: bool = False
    messages: List[SessionMessage] = field(default_factory=list)

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": self.cost,
            "summary_message_id": self.summary_message_id,
            "todos": [t.to_dict() for t in self.todos],
            "context": self.context,
            "parent_session_id": self.parent_session_id,
            "is_sub_agent_session": self.is_sub_agent_session,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Restore from a dictionary produced by ``to_dict()``."""
        return cls(
            session_id=data.get("session_id", uuid.uuid4().hex),
            title=data.get("title", "Untitled Session"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            message_count=data.get("message_count", 0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            cost=data.get("cost", 0.0),
            summary_message_id=data.get("summary_message_id"),
            todos=[TodoItem.from_dict(t) for t in data.get("todos", [])],
            context=data.get("context", {}),
            parent_session_id=data.get("parent_session_id"),
            is_sub_agent_session=data.get("is_sub_agent_session", False),
            messages=[
                SessionMessage.from_dict(m) for m in data.get("messages", [])
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Auto-Summarization Constants (Step 15.2)
# ═══════════════════════════════════════════════════════════════════════════

LARGE_CONTEXT_WINDOW_THRESHOLD = 200_000   # tokens — models with context > 200K
LARGE_CONTEXT_WINDOW_BUFFER = 20_000       # tokens — buffer for large context models
SMALL_CONTEXT_WINDOW_RATIO = 0.2           # summarize when 20% of context remains
DEFAULT_CONTEXT_WINDOW = 4_096             # default for unknown models (e.g., Ollama llama2)


# ═══════════════════════════════════════════════════════════════════════════
#  Summarization prompt template (Step 15.2)
# ═══════════════════════════════════════════════════════════════════════════

_SUMMARY_PROMPT = """\
You are summarizing a conversation to preserve context for continuing work later.

**Critical**: This summary will be the ONLY context available when the \
conversation resumes. Assume all previous messages will be lost. Be thorough.

**Required sections**:

## Current State
- What is the current state of the project/task?
- What files have been modified?
- What is working and what isn't?

## Files & Changes
- List all files that were created, modified, or deleted
- Note the purpose of each change

## Technical Context
- What languages, frameworks, tools are being used?
- What environment setup has been done?
- What dependencies were installed?

## Strategy & Approach
- What approach was taken?
- What alternatives were considered?

## Exact Next Steps
- What specific actions should be taken next?
- In what order?
"""

_TODO_SECTION_PREFIX = """
## Current Todo List
"""


# ═══════════════════════════════════════════════════════════════════════════
#  AgentSessionManager
# ═══════════════════════════════════════════════════════════════════════════

class AgentSessionManager:
    """Manages agent session persistence, loading, switching, and metadata.

    This class is part of the **stable orchestration layer**.  It does not
    embed any model-specific logic.  Any LLM integrated through the
    ``LLMRouter`` (Ollama, OpenAI, Gemini, etc.) is used dynamically via
    the ``_call_llm`` helper for title generation and summarization.

    Parameters
    ----------
    storage_dir : str
        Directory for session JSON files (relative to workspace root or
        absolute).  Defaults to ``.proxima/sessions``.
    """

    # Auto-save after this many messages are added
    _AUTO_SAVE_INTERVAL: int = 5

    def __init__(
        self,
        storage_dir: str = ".proxima/sessions",
        *,
        workspace_dir: Optional[str] = None,
    ) -> None:
        if workspace_dir is not None:
            # Per-workspace session isolation
            self._storage_dir = Path(workspace_dir) / ".proxima" / "sessions"
        else:
            self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._sessions: Dict[str, SessionState] = {}
        self._current_session_id: Optional[str] = None

        # Counter to track messages since last save
        self._unsaved_message_count: int = 0

        # Lightweight session-listing cache: maps session_id -> metadata dict
        self._listing_cache: Dict[str, Dict[str, Any]] = {}
        self._listing_cache_mtime: float = 0.0

    # ── Properties ────────────────────────────────────────────────────

    @property
    def storage_dir(self) -> Path:
        """The directory where session files are stored."""
        return self._storage_dir

    @property
    def current_session_id(self) -> Optional[str]:
        """The session ID of the currently active session."""
        return self._current_session_id

    # ══════════════════════════════════════════════════════════════════
    #  Step 15.1 — Core CRUD Operations
    # ══════════════════════════════════════════════════════════════════

    def create_session(
        self,
        title: str = "Untitled Session",
        *,
        parent_session_id: Optional[str] = None,
        is_sub_agent: bool = False,
    ) -> SessionState:
        """Create a new session, set it as current, and persist to disk.

        Parameters
        ----------
        title : str
            Human-readable session title.
        parent_session_id : str or None
            If this is a sub-agent session, the parent session ID.
        is_sub_agent : bool
            Mark as a sub-agent (restricted) session.

        Returns
        -------
        SessionState
            The newly created session.
        """
        # Persist the outgoing session before switching
        if self._current_session_id is not None:
            self._save_session(self._current_session_id)

        ctx_dict: Dict[str, Any] = {}
        if _SESSION_CONTEXT_AVAILABLE:
            ctx_dict = SessionContext().to_dict()

        session = SessionState(
            title=title,
            context=ctx_dict,
            parent_session_id=parent_session_id,
            is_sub_agent_session=is_sub_agent,
        )
        self._sessions[session.session_id] = session
        self._current_session_id = session.session_id
        self._unsaved_message_count = 0
        self._save_session(session.session_id)
        logger.info("Session created: %s (%s)", session.session_id[:8], title)
        return session

    def load_session(self, session_id: str) -> SessionState:
        """Load a session from cache or disk.

        Returns the session from the in-memory cache if already loaded,
        otherwise reads from disk.

        Raises
        ------
        FileNotFoundError
            If the session JSON file does not exist.
        """
        if session_id in self._sessions:
            return self._sessions[session_id]

        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        session = SessionState.from_dict(data)
        self._sessions[session.session_id] = session
        return session

    def switch_session(self, session_id: str) -> SessionState:
        """Save the current session, load and activate the requested one.

        Returns
        -------
        SessionState
            The newly active session.
        """
        # Persist the current session first
        if self._current_session_id is not None:
            self._save_session(self._current_session_id)

        session = self.load_session(session_id)
        self._current_session_id = session.session_id
        self._unsaved_message_count = 0
        logger.info("Switched to session: %s (%s)", session_id[:8], session.title)
        return session

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with lightweight metadata (no full messages).

        Returns a list sorted by ``updated_at`` descending (most recent first).
        Uses a lightweight cache keyed on file modification times to avoid
        re-reading every JSON file on each call.
        """
        # Build a quick mtime map for all session files
        current_files: Dict[str, float] = {}
        for path in self._storage_dir.glob("*.json"):
            try:
                current_files[path.stem] = path.stat().st_mtime
            except OSError:
                pass

        # Determine if cache is still valid
        cache_valid = (
            set(current_files.keys()) == set(self._listing_cache.keys())
            and all(
                current_files.get(sid, 0) <= self._listing_cache.get(sid, {}).get("_mtime", 0)
                for sid in current_files
            )
        )

        if not cache_valid:
            # Rebuild cache
            self._listing_cache.clear()
            for path in self._storage_dir.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    entry = {
                        "session_id": data.get("session_id", path.stem),
                        "title": data.get("title", "Untitled"),
                        "created_at": data.get("created_at", 0),
                        "updated_at": data.get("updated_at", 0),
                        "message_count": data.get("message_count", 0),
                        "is_sub_agent_session": data.get("is_sub_agent_session", False),
                        "_mtime": current_files.get(path.stem, 0),
                    }
                    self._listing_cache[entry["session_id"]] = entry
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Skipping corrupt session file %s: %s", path.name, exc)

        sessions = [
            {k: v for k, v in entry.items() if k != "_mtime"}
            for entry in self._listing_cache.values()
        ]
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> None:
        """Delete a session from both cache and disk."""
        self._sessions.pop(session_id, None)
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
        if self._current_session_id == session_id:
            self._current_session_id = None
        logger.info("Deleted session: %s", session_id[:8])

    # ══════════════════════════════════════════════════════════════════
    #  Session Import (Step 15.1)
    # ══════════════════════════════════════════════════════════════════

    def import_session(self, file_path: str) -> SessionState:
        """Import a session from an exported JSON file.

        Supports two export formats:

        1. ``AgentChatSession`` format (from ``_export_chat``):
           ``{id, name, messages: [{role, content, ...}], provider, model}``

        2. Legacy ``ai_conversation`` format:
           ``{timestamp, conversation: [...], execution_context: {...}}``

        Imported context (mentioned paths, repos, packages, branches) is
        extracted and populated into the new ``SessionContext`` so subsequent
        agent operations can reference them.

        Returns
        -------
        SessionState
            The newly created session with imported messages.
        """
        raw = json.loads(Path(file_path).read_text(encoding="utf-8"))

        messages: List[SessionMessage] = []
        title = "Imported Session"

        # ── Format 1: AgentChatSession ────────────────────────────────
        if "messages" in raw and isinstance(raw.get("messages"), list):
            title = raw.get("name", raw.get("title", "Imported Session"))
            for msg_data in raw["messages"]:
                messages.append(SessionMessage(
                    role=msg_data.get("role", "user"),
                    content=msg_data.get("content", ""),
                    timestamp=self._parse_timestamp(
                        msg_data.get("timestamp", ""),
                    ),
                    model=msg_data.get("model"),
                    tool_calls=msg_data.get("tool_calls", []),
                    tool_results=msg_data.get("tool_results", []),
                    metadata={
                        "tokens": msg_data.get("tokens", 0),
                        "thinking_time_ms": msg_data.get("thinking_time_ms", 0),
                    },
                ))

        # ── Format 2: Legacy ai_conversation ──────────────────────────
        elif "conversation" in raw and isinstance(raw.get("conversation"), list):
            ts_str = raw.get("timestamp", "")
            title = f"Imported ({ts_str})" if ts_str else "Imported Session"
            for item in raw["conversation"]:
                if isinstance(item, dict):
                    messages.append(SessionMessage(
                        role=item.get("role", "user"),
                        content=item.get("content", str(item)),
                    ))
                elif isinstance(item, str):
                    messages.append(SessionMessage(role="user", content=item))

        # ── Build SessionContext from imported messages ────────────────
        ctx_dict = self._extract_context_from_messages(messages)

        session = SessionState(
            title=title,
            messages=messages,
            message_count=len(messages),
            context=ctx_dict,
        )
        self._sessions[session.session_id] = session
        self._current_session_id = session.session_id
        self._save_session(session.session_id)
        logger.info(
            "Imported session from %s: %s (%d messages)",
            file_path, title, len(messages),
        )
        return session

    # ══════════════════════════════════════════════════════════════════
    #  Message Management (Step 15.1)
    # ══════════════════════════════════════════════════════════════════

    def add_message(
        self,
        message: SessionMessage,
        *,
        llm_router: Any = None,
    ) -> None:
        """Append a message to the current session.

        Auto-creates a session if none is active.  Triggers auto-save every
        ``_AUTO_SAVE_INTERVAL`` messages.

        On the **first user message** of a new session, auto-generates a
        title (Step 15.5).

        Parameters
        ----------
        message : SessionMessage
            The message to add.
        llm_router : optional
            If provided, used for auto-title generation on the first message.
        """
        session = self.get_current_session()
        if session is None:
            session = self.create_session()

        # Step 15.5: auto-generate title on first user message
        was_empty = session.message_count == 0
        has_default_title = session.title in ("Untitled Session", "")

        session.messages.append(message)
        session.message_count = len(session.messages)
        session.updated_at = time.time()

        self._unsaved_message_count += 1
        if self._unsaved_message_count >= self._AUTO_SAVE_INTERVAL:
            self._save_session(session.session_id)
            self._unsaved_message_count = 0

        # Auto-generate title only when the session has no explicit title
        if was_empty and has_default_title and message.role == "user" and message.content.strip():
            self.generate_title(llm_router, session.session_id)

    def get_current_session(self) -> Optional[SessionState]:
        """Return the currently active session, or ``None``."""
        if self._current_session_id is None:
            return None
        return self._sessions.get(self._current_session_id)

    # ══════════════════════════════════════════════════════════════════
    #  Step 15.2 — Auto-Summarization
    # ══════════════════════════════════════════════════════════════════

    def should_summarize(self, model_context_window: int = 0) -> bool:
        """Return ``True`` if the current conversation should be summarized.

        The decision is based on the estimated token count of messages that
        would be sent to the LLM in the next request, compared against the
        model's context window size.

        Parameters
        ----------
        model_context_window : int
            The context window size of the model in tokens.  If zero or
            negative, falls back to ``DEFAULT_CONTEXT_WINDOW``.
        """
        if model_context_window <= 0:
            model_context_window = DEFAULT_CONTEXT_WINDOW
        session = self.get_current_session()
        if session is None:
            return False

        messages_for_llm = self.get_messages_for_llm()
        if not messages_for_llm:
            return False

        # Estimate tokens: ~1 token per 4 characters (heuristic)
        current_tokens = sum(len(m.content) // 4 for m in messages_for_llm)
        remaining = model_context_window - current_tokens

        if model_context_window > LARGE_CONTEXT_WINDOW_THRESHOLD:
            return remaining <= LARGE_CONTEXT_WINDOW_BUFFER
        else:
            return remaining <= model_context_window * SMALL_CONTEXT_WINDOW_RATIO

    def summarize_session(
        self,
        llm_router: Any,
        session_id: Optional[str] = None,
    ) -> Optional[SessionMessage]:
        """Summarize the session's conversation via the LLM.

        Replaces all prior context with a single summary message.  If the
        session has already been summarized, only messages *after* the last
        summary are included.

        Parameters
        ----------
        llm_router
            An ``LLMRouter`` instance (or compatible) used to generate the
            summary.  If ``None`` or unavailable, returns ``None``.
        session_id : str or None
            Session to summarize.  Defaults to the current session.

        Returns
        -------
        SessionMessage or None
            The summary message, or ``None`` if summarization was skipped.
        """
        sid = session_id or self._current_session_id
        if sid is None:
            return None
        session = self._sessions.get(sid)
        if session is None:
            return None
        if llm_router is None:
            return None

        # Determine which messages to include in summarization
        start_idx = 0
        if session.summary_message_id:
            for idx, msg in enumerate(session.messages):
                if msg.message_id == session.summary_message_id:
                    start_idx = idx + 1
                    break

        msgs_to_summarize = session.messages[start_idx:]
        if len(msgs_to_summarize) < 3:
            # Not enough content to warrant summarization
            return None

        # Build the summarization prompt
        prompt = _SUMMARY_PROMPT

        # Append todo section if there are active todos
        if session.todos:
            prompt += _TODO_SECTION_PREFIX
            for todo in session.todos:
                status_marker = todo.status
                prompt += f"- [{status_marker}] {todo.content}\n"
            prompt += (
                "\nInclude these tasks and their statuses in your summary.\n"
                "Instruct the resuming assistant to use the todos tool to "
                "continue tracking progress on these tasks.\n"
            )

        # Build conversation text for the LLM
        conversation_text = "\n\n".join(
            f"**{m.role.upper()}** ({m.message_id[:8]}):\n{m.content}"
            for m in msgs_to_summarize
            if m.content.strip()
        )

        full_prompt = (
            f"{prompt}\n\n"
            f"--- CONVERSATION TO SUMMARIZE ---\n\n"
            f"{conversation_text}\n\n"
            f"--- END CONVERSATION ---\n\n"
            f"Write your summary now:"
        )

        # Call the LLM (model-agnostic — works with any integrated model)
        try:
            summary_text = self._call_llm(llm_router, full_prompt)
        except Exception as exc:
            logger.error("Session summarization failed: %s", exc)
            return None

        if not summary_text or not summary_text.strip():
            return None

        # Strip think tags if present
        summary_text = self._strip_think_tags(summary_text)

        # Create the summary message
        summary_msg = SessionMessage(
            role="assistant",
            content=summary_text,
            is_summary=True,
            model="auto-summary",
            metadata={"summarized_message_count": len(msgs_to_summarize)},
        )

        session.messages.append(summary_msg)
        session.summary_message_id = summary_msg.message_id
        session.message_count = len(session.messages)

        # Accumulate token counters — preserve lifetime totals across
        # summarizations so we don't lose historical usage data.
        session.prompt_tokens += sum(
            len(m.content) // 4 for m in msgs_to_summarize if m.role == "user"
        )
        session.completion_tokens += len(summary_text) // 4

        self._save_session(session.session_id)
        logger.info(
            "Session %s summarized: %d messages → summary (%d chars)",
            sid[:8], len(msgs_to_summarize), len(summary_text),
        )
        return summary_msg

    def get_messages_for_llm(
        self,
        session_id: Optional[str] = None,
    ) -> List[SessionMessage]:
        """Return the messages that should be sent to the LLM.

        If the session has been summarized, returns the summary message plus
        all subsequent messages.  Otherwise returns all messages.

        Tool results in returned messages are truncated to 500 characters
        to reduce context usage while preserving full results in storage.
        """
        sid = session_id or self._current_session_id
        if sid is None:
            return []
        session = self._sessions.get(sid)
        if session is None:
            return []

        messages = session.messages
        start_idx = 0

        if session.summary_message_id:
            for idx, msg in enumerate(messages):
                if msg.message_id == session.summary_message_id:
                    start_idx = idx
                    break

        result: List[SessionMessage] = []
        for msg in messages[start_idx:]:
            # Create a copy with truncated tool results for context efficiency
            truncated = SessionMessage(
                message_id=msg.message_id,
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
                model=msg.model,
                tool_calls=msg.tool_calls,
                tool_results=[
                    {**tr, "output": tr.get("output", "")[:500]}
                    for tr in msg.tool_results
                ] if msg.tool_results else [],
                is_summary=msg.is_summary,
                metadata=msg.metadata,
            )
            result.append(truncated)
        return result

    # ══════════════════════════════════════════════════════════════════
    #  Step 15.5 — Title Auto-Generation
    # ══════════════════════════════════════════════════════════════════

    def generate_title(
        self,
        llm_router: Any,
        session_id: Optional[str] = None,
    ) -> str:
        """Auto-generate a concise session title from the first user message.

        If an LLM is available, asks it to generate a max-8-word title.
        Otherwise falls back to the first 8 words of the first message.

        Parameters
        ----------
        llm_router
            ``LLMRouter`` or compatible.  May be ``None``.
        session_id : str or None
            Target session.  Defaults to the current session.

        Returns
        -------
        str
            The generated (or fallback) title.
        """
        sid = session_id or self._current_session_id
        if sid is None:
            return "Untitled Session"
        session = self._sessions.get(sid)
        if session is None:
            return "Untitled Session"

        # Find the first user message
        first_user_msg = ""
        for msg in session.messages:
            if msg.role == "user" and msg.content.strip():
                first_user_msg = msg.content.strip()
                break
        if not first_user_msg:
            return "Untitled Session"

        # Try LLM-based title generation (model-agnostic)
        if llm_router is not None:
            try:
                prompt = (
                    f"Generate a concise title (max 8 words) for this "
                    f"conversation: {first_user_msg[:500]}"
                )
                title = self._call_llm(llm_router, prompt, max_tokens=40)
                if title:
                    title = self._strip_think_tags(title).strip().strip('"\'')
                    if title:
                        session.title = title
                        self._save_session(session.session_id)
                        return title
            except Exception:
                logger.debug("LLM title generation failed, using fallback")

        # Fallback: first 8 words
        words = first_user_msg.split()[:8]
        title = " ".join(words)
        if len(first_user_msg.split()) > 8:
            title += "..."
        session.title = title
        self._save_session(session.session_id)
        return title

    # ══════════════════════════════════════════════════════════════════
    #  Todo Management
    # ══════════════════════════════════════════════════════════════════

    def add_todo(self, content: str, active_form: str = "") -> Optional[TodoItem]:
        """Add a todo item to the current session.

        Parameters
        ----------
        content : str
            Imperative-form description (e.g. "Run tests").
        active_form : str
            Present-continuous form (e.g. "Running tests").
            Defaults to the content if not provided.

        Returns
        -------
        TodoItem or None
            The created todo, or ``None`` if no active session.
        """
        session = self.get_current_session()
        if session is None:
            return None
        todo = TodoItem(
            content=content,
            active_form=active_form or content,
        )
        session.todos.append(todo)
        return todo

    def update_todo_status(self, index: int, status: str) -> bool:
        """Update the status of a todo item by index.

        Enforces exactly one ``in_progress`` todo at a time — if setting a
        todo to ``in_progress``, all others with that status are moved back
        to ``pending``.

        Parameters
        ----------
        index : int
            Zero-based index into the session's todo list.
        status : str
            New status: ``"pending"``, ``"in_progress"``, or ``"completed"``.

        Returns
        -------
        bool
            ``True`` if the update succeeded.
        """
        session = self.get_current_session()
        if session is None or index < 0 or index >= len(session.todos):
            return False

        if status == "in_progress":
            # Enforce single in-progress constraint
            for todo in session.todos:
                if todo.status == "in_progress":
                    todo.status = "pending"

        session.todos[index].status = status
        return True

    def get_todos(self) -> List[TodoItem]:
        """Return all todos for the current session."""
        session = self.get_current_session()
        if session is None:
            return []
        return list(session.todos)

    def get_todo_progress(self) -> Tuple[int, int]:
        """Return ``(completed_count, total_count)``."""
        session = self.get_current_session()
        if session is None:
            return (0, 0)
        completed = sum(1 for t in session.todos if t.status == "completed")
        return (completed, len(session.todos))

    # ══════════════════════════════════════════════════════════════════
    #  SessionContext Bridge
    # ══════════════════════════════════════════════════════════════════

    def get_session_context(self) -> Optional["SessionContext"]:
        """Reconstruct a ``SessionContext`` from the stored dict.

        Returns ``None`` if ``SessionContext`` is not importable or no
        session is active.
        """
        if not _SESSION_CONTEXT_AVAILABLE:
            return None
        session = self.get_current_session()
        if session is None:
            return None
        return SessionContext.from_dict(session.context)

    def save_session_context(self, context: "SessionContext") -> None:
        """Serialize and store the current ``SessionContext`` into the session."""
        session = self.get_current_session()
        if session is None:
            return
        if _SESSION_CONTEXT_AVAILABLE:
            session.context = context.to_dict()

    def save_current(self) -> None:
        """Explicitly save the current session to disk (flush)."""
        if self._current_session_id:
            self._save_session(self._current_session_id)

    def load_most_recent_session(self) -> Optional[SessionState]:
        """Load and activate the most recently updated session.

        Useful on app startup to resume the last session automatically.

        Returns
        -------
        SessionState or None
            The restored session, or ``None`` if no sessions exist.
        """
        sessions = self.list_sessions()
        if not sessions:
            return None
        most_recent = sessions[0]  # already sorted by updated_at desc
        return self.switch_session(most_recent["session_id"])

    # ══════════════════════════════════════════════════════════════════
    #  Internal Helpers
    # ══════════════════════════════════════════════════════════════════

    def _session_path(self, session_id: str) -> Path:
        """Return the filesystem path for a session file."""
        return self._storage_dir / f"{session_id}.json"

    def _save_session(self, session_id: str) -> None:
        """Atomically persist a session to JSON on disk.

        Writes to a ``.tmp`` file first, then renames to prevent corruption
        from interrupted writes.  Uses platform-appropriate file locking to
        guard against concurrent access from multiple processes.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return

        data = session.to_dict()
        tmp_path = self._session_path(session_id).with_suffix(".tmp")
        final_path = self._session_path(session_id)

        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                # Acquire an exclusive file lock (platform-specific)
                self._lock_file(fh)
                try:
                    fh.write(json.dumps(data, indent=2, default=str))
                    fh.flush()
                    os.fsync(fh.fileno())
                finally:
                    self._unlock_file(fh)

            # Atomic rename (same filesystem)
            tmp_path.replace(final_path)

            # Invalidate listing cache for this session
            self._listing_cache.pop(session_id, None)
        except OSError as exc:
            logger.error("Failed to save session %s: %s", session_id[:8], exc)
            # Clean up tmp file if rename failed
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _lock_file(fh: Any) -> None:
        """Acquire an exclusive file lock (best-effort, platform-specific)."""
        try:
            if _HAS_MSVCRT:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            elif _HAS_FCNTL:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError):
            pass  # Lock not acquired — proceed anyway (single-user is safe)

    @staticmethod
    def _unlock_file(fh: Any) -> None:
        """Release the file lock (best-effort)."""
        try:
            if _HAS_MSVCRT:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            elif _HAS_FCNTL:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except (OSError, IOError):
            pass

    def _extract_context_from_messages(
        self,
        messages: List[SessionMessage],
    ) -> Dict[str, Any]:
        """Scan imported messages and build a ``SessionContext`` dict.

        Extracts URLs, file paths, branches, and packages mentioned in the
        conversation so subsequent operations can reference them.
        """
        paths: List[str] = []
        urls: List[str] = []
        branches: List[str] = []
        packages: List[str] = []

        _url_re = re.compile(r'https?://\S+')
        # Windows and Unix path patterns
        _path_re = re.compile(
            r'(?:[A-Za-z]:\\[\w\\.\-/ ]+|'
            r'/(?:home|usr|var|tmp|opt|etc|mnt|Users)[\w/.\- ]+)'
        )
        _branch_re = re.compile(
            r'(?:branch|checkout|switch)\s+(?:to\s+)?([a-zA-Z0-9_./-]+)',
            re.IGNORECASE,
        )
        _pkg_re = re.compile(
            r'pip\s+install\s+([\w\->=<~!]+(?:\s+[\w\->=<~!]+)*)',
            re.IGNORECASE,
        )

        for msg in messages:
            text = msg.content
            if not text:
                continue

            # URLs
            for match in _url_re.finditer(text):
                url = match.group().rstrip(".,;:)")
                if url not in urls:
                    urls.append(url)
                    if len(urls) >= 10:
                        break

            # Paths
            for match in _path_re.finditer(text):
                p = match.group().rstrip(".,;:)")
                if p not in paths:
                    paths.append(p)
                    if len(paths) >= 10:
                        break

            # Branches
            for match in _branch_re.finditer(text):
                b = match.group(1)
                if b not in branches:
                    branches.append(b)
                    if len(branches) >= 10:
                        break

            # Packages
            for match in _pkg_re.finditer(text):
                for pkg in match.group(1).split():
                    pkg_name = re.split(r'[>=<~!]', pkg)[0]
                    if pkg_name and pkg_name not in packages:
                        packages.append(pkg_name)
                        if len(packages) >= 20:
                            break

        ctx_dict: Dict[str, Any] = {}
        if _SESSION_CONTEXT_AVAILABLE:
            ctx_dict = SessionContext().to_dict()

        ctx_dict["last_mentioned_paths"] = paths[:10]
        ctx_dict["last_mentioned_urls"] = urls[:10]
        ctx_dict["last_mentioned_branches"] = branches[:10]
        ctx_dict["last_mentioned_packages"] = packages[:10]
        ctx_dict["installed_packages"] = packages[:10]

        return ctx_dict

    @staticmethod
    def _parse_timestamp(value: Any) -> float:
        """Best-effort parsing of a timestamp string to a Unix float."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value:
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(value.replace("Z", "+00:00"))
                return dt.timestamp()
            except (ValueError, TypeError):
                pass
        return time.time()

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove ``<think>...</think>`` wrapper from LLM output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _call_llm(
        llm_router: Any,
        prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        """Call the LLM router with a simple single-message request.

        This is model-agnostic — it works with any model integrated through
        the ``LLMRouter`` (Ollama, OpenAI, Anthropic, Google, etc.).  The
        assistant architecture is stable; the model provides dynamic
        intelligence.
        """
        if not _LLM_AVAILABLE or llm_router is None:
            return ""

        try:
            request = LLMRequest(
                prompt=prompt,
                system_prompt="You are a helpful assistant.",
                max_tokens=max_tokens,
                temperature=0.3,
            )
            response = llm_router.route(request)
            if response and hasattr(response, "text"):
                return response.text or ""
            if response and isinstance(response, str):
                return response
        except Exception as exc:
            logger.debug("LLM call failed in session manager: %s", exc)
        return ""
