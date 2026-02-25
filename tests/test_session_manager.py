"""Tests for Phase 15: AgentSessionManager — session persistence, import,
auto-summarization, title generation, and context window management.

Covers the 7 required test cases from the implementation spec (Step 15.6).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from proxima.agent.agent_session_manager import (
    AgentSessionManager,
    SessionMessage,
    SessionState,
    TodoItem,
    LARGE_CONTEXT_WINDOW_BUFFER,
    LARGE_CONTEXT_WINDOW_THRESHOLD,
    SMALL_CONTEXT_WINDOW_RATIO,
    DEFAULT_CONTEXT_WINDOW,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for session storage."""
    return tmp_path / "sessions"


@pytest.fixture()
def mgr(tmp_dir: Path) -> AgentSessionManager:
    """Return a fresh AgentSessionManager backed by a temp directory."""
    return AgentSessionManager(storage_dir=str(tmp_dir))


@pytest.fixture()
def mock_llm_router() -> MagicMock:
    """Return a mock LLMRouter that returns predictable text."""
    router = MagicMock()
    response = MagicMock()
    response.text = "Test title for conversation"
    router.route.return_value = response
    return router


# ── 1. Session Persistence ─────────────────────────────────────────

class TestSessionPersistence:
    """Create a session, add messages, save, reload, verify everything."""

    def test_create_and_reload(self, mgr: AgentSessionManager) -> None:
        session = mgr.create_session(title="Persist Test")
        sid = session.session_id

        # Add messages (llm_router=None so title fallback = first words)
        mgr.add_message(SessionMessage(role="user", content="Hello"))
        # Restore the explicit title after auto-title overwrites it
        session.title = "Persist Test"
        mgr.add_message(SessionMessage(role="assistant", content="Hi there"))
        mgr.add_message(SessionMessage(role="tool", content="result", tool_results=[{"tool_name": "ls"}]))

        # Force save
        mgr._save_session(sid)

        # Remove from in-memory cache to force disk load
        del mgr._sessions[sid]

        loaded = mgr.load_session(sid)
        assert loaded.session_id == sid
        assert loaded.title == "Persist Test"
        assert loaded.message_count == 3
        assert len(loaded.messages) == 3
        assert loaded.messages[0].role == "user"
        assert loaded.messages[0].content == "Hello"
        assert loaded.messages[1].role == "assistant"
        assert loaded.messages[2].tool_results == [{"tool_name": "ls"}]

    def test_timestamps_preserved(self, mgr: AgentSessionManager) -> None:
        session = mgr.create_session(title="Timestamp Test")
        created = session.created_at
        time.sleep(0.05)
        mgr.add_message(SessionMessage(role="user", content="a"))

        mgr._save_session(session.session_id)
        del mgr._sessions[session.session_id]

        loaded = mgr.load_session(session.session_id)
        assert loaded.created_at == created
        assert loaded.updated_at >= created

    def test_auto_save_interval(self, mgr: AgentSessionManager) -> None:
        """After _AUTO_SAVE_INTERVAL messages, the session auto-saves."""
        session = mgr.create_session(title="Auto-Save Test")
        sid = session.session_id

        for i in range(mgr._AUTO_SAVE_INTERVAL):
            mgr.add_message(SessionMessage(role="user", content=f"msg {i}"))

        # File should exist on disk after AUTO_SAVE_INTERVAL messages
        path = mgr._session_path(sid)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["message_count"] >= mgr._AUTO_SAVE_INTERVAL


# ── 2. Session Listing and Switching ─────────────────────────────────

class TestSessionListingAndSwitching:
    """Create 3 sessions, list, switch between them, verify active session."""

    def test_list_sessions(self, mgr: AgentSessionManager) -> None:
        s1 = mgr.create_session(title="First")
        s2 = mgr.create_session(title="Second")
        s3 = mgr.create_session(title="Third")

        listing = mgr.list_sessions()
        titles = {s["title"] for s in listing}
        assert "First" in titles
        assert "Second" in titles
        assert "Third" in titles
        assert len(listing) >= 3

    def test_switch_session(self, mgr: AgentSessionManager) -> None:
        s1 = mgr.create_session(title="Alpha")
        s2 = mgr.create_session(title="Beta")

        # Currently on Beta
        assert mgr.current_session_id == s2.session_id

        # Switch to Alpha
        loaded = mgr.switch_session(s1.session_id)
        assert loaded.title == "Alpha"
        assert mgr.current_session_id == s1.session_id

    def test_listing_sorted_by_updated(self, mgr: AgentSessionManager) -> None:
        s1 = mgr.create_session(title="Old")
        time.sleep(0.05)
        s2 = mgr.create_session(title="New")

        listing = mgr.list_sessions()
        # New should be first (most recently updated)
        assert listing[0]["title"] == "New"


# ── 3. Session Import ─────────────────────────────────────────────────

class TestSessionImport:
    """Import from AgentChatSession and legacy formats."""

    def test_import_agent_chat_format(self, mgr: AgentSessionManager, tmp_path: Path) -> None:
        export_data = {
            "id": "test_chat_001",
            "name": "Exported Chat",
            "messages": [
                {"role": "user", "content": "Clone the repo", "timestamp": "2025-01-29T15:34:00"},
                {"role": "assistant", "content": "Cloning now...", "model": "ollama/qwen"},
            ],
            "provider": "ollama",
            "model": "qwen2.5-coder:7b",
        }
        export_file = tmp_path / "export.json"
        export_file.write_text(json.dumps(export_data), encoding="utf-8")

        session = mgr.import_session(str(export_file))
        assert session.title == "Exported Chat"
        assert session.message_count == 2
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Clone the repo"
        assert session.messages[1].model == "ollama/qwen"

    def test_import_legacy_format(self, mgr: AgentSessionManager, tmp_path: Path) -> None:
        export_data = {
            "timestamp": "2025-01-29 20:41:17",
            "conversation": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }
        export_file = tmp_path / "legacy.json"
        export_file.write_text(json.dumps(export_data), encoding="utf-8")

        session = mgr.import_session(str(export_file))
        assert session.message_count == 2
        assert "Imported" in session.title
        assert session.messages[0].content == "hello"

    def test_import_with_context_extraction(self, mgr: AgentSessionManager, tmp_path: Path) -> None:
        export_data = {
            "id": "ctx_test",
            "name": "Context Extraction",
            "messages": [
                {"role": "user", "content": "Clone https://github.com/test/repo and checkout main branch"},
                {"role": "assistant", "content": "Done. Installed numpy and pandas via pip."},
            ],
        }
        export_file = tmp_path / "ctx.json"
        export_file.write_text(json.dumps(export_data), encoding="utf-8")

        session = mgr.import_session(str(export_file))
        ctx = session.context
        # The context extraction should have found URLs and packages
        assert isinstance(ctx, dict)


# ── 4. Session Deletion ───────────────────────────────────────────────

class TestSessionDeletion:
    """Create, delete, verify removed from listing and disk."""

    def test_delete_removes_from_listing(self, mgr: AgentSessionManager) -> None:
        s = mgr.create_session(title="Doomed")
        sid = s.session_id
        mgr.delete_session(sid)

        listing = mgr.list_sessions()
        ids = {s["session_id"] for s in listing}
        assert sid not in ids

    def test_delete_removes_file(self, mgr: AgentSessionManager) -> None:
        s = mgr.create_session(title="Doomed2")
        sid = s.session_id
        path = mgr._session_path(sid)
        assert path.exists()

        mgr.delete_session(sid)
        assert not path.exists()

    def test_delete_clears_current(self, mgr: AgentSessionManager) -> None:
        s = mgr.create_session(title="Current")
        assert mgr.current_session_id == s.session_id

        mgr.delete_session(s.session_id)
        assert mgr.current_session_id is None


# ── 5. Auto-Summarization Trigger ────────────────────────────────────

class TestAutoSummarizationTrigger:
    """Verify should_summarize() triggers at the right thresholds."""

    def _fill_session(self, mgr: AgentSessionManager, char_count: int) -> None:
        """Add messages totalling approximately *char_count* chars."""
        session = mgr.get_current_session()
        if session is None:
            session = mgr.create_session()
        # Each message ~200 chars
        num_messages = max(1, char_count // 200)
        for i in range(num_messages):
            mgr.add_message(SessionMessage(
                role="user" if i % 2 == 0 else "assistant",
                content="x" * 200,
            ))

    def test_small_context_window_triggers(self, mgr: AgentSessionManager) -> None:
        """With a small context window (4096 tokens ≈ 16384 chars), filling
        past 80% should trigger summarization."""
        mgr.create_session()
        small_window = 4096  # tokens
        # Fill with ~80% of the tokens (4096 * 4 * 0.85 ≈ 13926 chars)
        self._fill_session(mgr, int(small_window * 4 * 0.85))
        assert mgr.should_summarize(small_window) is True

    def test_small_context_not_triggered_below_threshold(self, mgr: AgentSessionManager) -> None:
        """With a small window, being at 30% should NOT trigger."""
        mgr.create_session()
        small_window = 4096
        self._fill_session(mgr, int(small_window * 4 * 0.3))
        assert mgr.should_summarize(small_window) is False

    def test_large_context_window_triggers(self, mgr: AgentSessionManager) -> None:
        """With a large context window (>200K tokens), trigger only when
        remaining is less than LARGE_CONTEXT_WINDOW_BUFFER."""
        mgr.create_session()
        large_window = LARGE_CONTEXT_WINDOW_THRESHOLD + 50_000
        # Fill almost all tokens: total - buffer + some extra
        fill_tokens = large_window - LARGE_CONTEXT_WINDOW_BUFFER + 5000
        self._fill_session(mgr, fill_tokens * 4)
        assert mgr.should_summarize(large_window) is True

    def test_large_context_not_triggered_plenty_room(self, mgr: AgentSessionManager) -> None:
        """Large window with plenty of room should NOT trigger."""
        mgr.create_session()
        large_window = LARGE_CONTEXT_WINDOW_THRESHOLD + 50_000
        self._fill_session(mgr, 2000)  # tiny amount
        assert mgr.should_summarize(large_window) is False

    def test_summarize_produces_summary_message(
        self, mgr: AgentSessionManager, mock_llm_router: MagicMock,
    ) -> None:
        """Verify that summarize_session produces a valid summary message."""
        mock_llm_router.route.return_value.text = "This is a session summary."

        session = mgr.create_session()
        for i in range(5):
            mgr.add_message(SessionMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message content number {i} with some detail " * 5,
            ))

        result = mgr.summarize_session(mock_llm_router, session.session_id)
        assert result is not None
        assert result.is_summary is True
        assert result.content == "This is a session summary."
        assert session.summary_message_id == result.message_id


# ── 6. Title Auto-Generation ─────────────────────────────────────────

class TestTitleAutoGeneration:
    """Verify titles are generated on first message."""

    def test_llm_title_generation(
        self, mgr: AgentSessionManager, mock_llm_router: MagicMock,
    ) -> None:
        mock_llm_router.route.return_value.text = "Quantum Circuit Simulation Setup"

        session = mgr.create_session()
        assert session.title == "Untitled Session"

        mgr.add_message(
            SessionMessage(role="user", content="Set up a quantum circuit simulation"),
            llm_router=mock_llm_router,
        )

        # Title should now be updated
        assert session.title == "Quantum Circuit Simulation Setup"

    def test_fallback_title_without_llm(self, mgr: AgentSessionManager) -> None:
        session = mgr.create_session()
        mgr.add_message(
            SessionMessage(role="user", content="Hello world how are you doing today my friend"),
            llm_router=None,
        )
        # Should fall back to first 8 words
        assert session.title != "Untitled Session"
        assert len(session.title.split()) <= 9  # 8 words + possible "..."

    def test_title_not_regenerated_on_second_message(
        self, mgr: AgentSessionManager, mock_llm_router: MagicMock,
    ) -> None:
        mock_llm_router.route.return_value.text = "First Title"
        session = mgr.create_session()

        mgr.add_message(
            SessionMessage(role="user", content="first message"),
            llm_router=mock_llm_router,
        )
        assert session.title == "First Title"

        mock_llm_router.route.return_value.text = "Should Not Be Used"
        mgr.add_message(
            SessionMessage(role="user", content="second message"),
            llm_router=mock_llm_router,
        )
        # Title should stay as "First Title"
        assert session.title == "First Title"


# ── 7. Context Window Calculation ────────────────────────────────────

class TestContextWindowCalculation:
    """Verify get_messages_for_llm returns correct subsets."""

    def test_all_messages_without_summary(self, mgr: AgentSessionManager) -> None:
        session = mgr.create_session()
        for i in range(5):
            mgr.add_message(SessionMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"msg {i}",
            ))

        msgs = mgr.get_messages_for_llm()
        assert len(msgs) == 5
        assert msgs[0].content == "msg 0"
        assert msgs[4].content == "msg 4"

    def test_messages_from_summary_onward(
        self, mgr: AgentSessionManager, mock_llm_router: MagicMock,
    ) -> None:
        mock_llm_router.route.return_value.text = "Session summary text."
        session = mgr.create_session()

        for i in range(5):
            mgr.add_message(SessionMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"old msg {i}" + " " * 200,
            ))

        mgr.summarize_session(mock_llm_router, session.session_id)

        # Add new messages after summarization
        mgr.add_message(SessionMessage(role="user", content="new question"))
        mgr.add_message(SessionMessage(role="assistant", content="new answer"))

        msgs = mgr.get_messages_for_llm()
        # Should start from the summary message
        assert msgs[0].is_summary is True
        assert msgs[0].content == "Session summary text."
        # Plus the 2 new messages after summary
        assert any(m.content == "new question" for m in msgs)
        assert any(m.content == "new answer" for m in msgs)

    def test_tool_results_truncated(self, mgr: AgentSessionManager) -> None:
        session = mgr.create_session()
        long_output = "x" * 2000
        mgr.add_message(SessionMessage(
            role="tool",
            content="tool ran",
            tool_results=[{"tool_name": "test", "output": long_output}],
        ))

        msgs = mgr.get_messages_for_llm()
        assert len(msgs) == 1
        # Tool result output should be truncated to 500 chars
        assert len(msgs[0].tool_results[0]["output"]) == 500


# ── TodoItem dataclass ─────────────────────────────────────────────

class TestTodoItem:
    """Basic tests for the TodoItem dataclass serialization."""

    def test_round_trip(self) -> None:
        todo = TodoItem(content="Run tests", status="in_progress", active_form="Running tests")
        data = todo.to_dict()
        restored = TodoItem.from_dict(data)
        assert restored.content == "Run tests"
        assert restored.status == "in_progress"
        assert restored.active_form == "Running tests"


# ── SessionMessage dataclass ──────────────────────────────────────

class TestSessionMessage:
    """Basic tests for SessionMessage serialization."""

    def test_round_trip(self) -> None:
        msg = SessionMessage(
            role="assistant",
            content="Hello",
            model="test-model",
            tool_calls=[{"name": "ls", "arguments": {}}],
            is_summary=True,
        )
        data = msg.to_dict()
        restored = SessionMessage.from_dict(data)
        assert restored.role == "assistant"
        assert restored.content == "Hello"
        assert restored.model == "test-model"
        assert restored.is_summary is True
        assert len(restored.tool_calls) == 1


# ── SessionState dataclass ────────────────────────────────────────

class TestSessionState:
    """Basic tests for SessionState serialization."""

    def test_round_trip(self) -> None:
        state = SessionState(
            title="Test Session",
            message_count=3,
            prompt_tokens=100,
            completion_tokens=50,
            todos=[TodoItem(content="Task 1")],
            messages=[
                SessionMessage(role="user", content="hi"),
                SessionMessage(role="assistant", content="hello"),
            ],
        )
        data = state.to_dict()
        restored = SessionState.from_dict(data)
        assert restored.title == "Test Session"
        assert restored.message_count == 3
        assert restored.prompt_tokens == 100
        assert len(restored.todos) == 1
        assert len(restored.messages) == 2
        assert restored.messages[0].content == "hi"


# ── Todo Management ──────────────────────────────────────────────

class TestTodoManagement:
    """Test add / update / list / progress methods."""

    def test_add_and_list(self, mgr: AgentSessionManager) -> None:
        mgr.create_session()
        mgr.add_todo("Write tests")
        mgr.add_todo("Review code")
        todos = mgr.get_todos()
        assert len(todos) == 2
        assert todos[0].content == "Write tests"
        assert todos[0].status == "pending"

    def test_update_status(self, mgr: AgentSessionManager) -> None:
        mgr.create_session()
        mgr.add_todo("Task A")
        mgr.update_todo_status(0, "completed")
        todos = mgr.get_todos()
        assert todos[0].status == "completed"

    def test_progress(self, mgr: AgentSessionManager) -> None:
        mgr.create_session()
        mgr.add_todo("T1")
        mgr.add_todo("T2")
        mgr.add_todo("T3")
        mgr.update_todo_status(0, "completed")
        mgr.update_todo_status(1, "in_progress")

        completed, total = mgr.get_todo_progress()
        assert total == 3
        assert completed == 1


# ── Load Most Recent Session ─────────────────────────────────────

class TestLoadMostRecent:
    """Verify load_most_recent_session returns the latest one."""

    def test_loads_most_recent(self, mgr: AgentSessionManager) -> None:
        s1 = mgr.create_session(title="Older")
        time.sleep(0.05)
        s2 = mgr.create_session(title="Newer")
        mgr._save_session(s2.session_id)

        # Clear cache
        mgr._sessions.clear()
        mgr._current_session_id = None

        loaded = mgr.load_most_recent_session()
        assert loaded is not None
        assert loaded.title == "Newer"
        assert mgr.current_session_id == loaded.session_id


# ── 8. Corrupt JSON Import ────────────────────────────────────────

class TestCorruptJsonImport:
    """Verify import_session handles corrupted JSON gracefully."""

    def test_corrupt_json_raises_value_error(self, mgr: AgentSessionManager, tmp_path: Path) -> None:
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("{this is not valid json!", encoding="utf-8")
        with pytest.raises((ValueError, json.JSONDecodeError)):
            mgr.import_session(str(bad_file))

    def test_empty_file_raises(self, mgr: AgentSessionManager, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises((ValueError, json.JSONDecodeError)):
            mgr.import_session(str(empty_file))

    def test_missing_file_raises(self, mgr: AgentSessionManager) -> None:
        with pytest.raises((FileNotFoundError, OSError)):
            mgr.import_session("/nonexistent/path/no_file.json")


# ── 9. Context Extraction Accuracy ───────────────────────────────

class TestContextExtractionAccuracy:
    """Verify context extraction captures URLs, packages, branches, and paths."""

    def test_url_extraction(self, mgr: AgentSessionManager, tmp_path: Path) -> None:
        export_data = {
            "id": "url_ctx",
            "name": "With URLs",
            "messages": [
                {"role": "user", "content": "Fetch https://example.com/api and https://github.com/org/repo"},
                {"role": "assistant", "content": "Fetched both URLs."},
            ],
        }
        export_file = tmp_path / "urls.json"
        export_file.write_text(json.dumps(export_data), encoding="utf-8")

        session = mgr.import_session(str(export_file))
        ctx = session.context
        assert isinstance(ctx, dict)
        urls = ctx.get("last_mentioned_urls", [])
        assert isinstance(urls, list)
        assert len(urls) >= 2, f"Expected at least 2 URLs extracted, got {urls}"
        assert any("example.com" in u for u in urls)
        assert any("github.com" in u for u in urls)

    def test_package_extraction(self, mgr: AgentSessionManager, tmp_path: Path) -> None:
        export_data = {
            "id": "pkg_ctx",
            "name": "With Packages",
            "messages": [
                {"role": "assistant", "content": "pip install numpy pandas scikit-learn"},
            ],
        }
        export_file = tmp_path / "pkgs.json"
        export_file.write_text(json.dumps(export_data), encoding="utf-8")

        session = mgr.import_session(str(export_file))
        ctx = session.context
        assert isinstance(ctx, dict)
        packages = ctx.get("installed_packages", [])
        assert isinstance(packages, list)
        assert "numpy" in packages, f"Expected 'numpy' in packages, got {packages}"
        assert "pandas" in packages, f"Expected 'pandas' in packages, got {packages}"

    def test_branch_extraction(self, mgr: AgentSessionManager, tmp_path: Path) -> None:
        export_data = {
            "id": "branch_ctx",
            "name": "With Branches",
            "messages": [
                {"role": "user", "content": "checkout to feature/quantum-sim"},
                {"role": "assistant", "content": "Switched to branch feature/quantum-sim."},
            ],
        }
        export_file = tmp_path / "branches.json"
        export_file.write_text(json.dumps(export_data), encoding="utf-8")

        session = mgr.import_session(str(export_file))
        ctx = session.context
        assert isinstance(ctx, dict)
        branches = ctx.get("last_mentioned_branches", [])
        assert isinstance(branches, list)
        assert any("feature/quantum-sim" in b for b in branches), (
            f"Expected 'feature/quantum-sim' in branches, got {branches}"
        )


class TestExtractContextFromMessagesUnit:
    """Direct unit tests for AgentSessionManager._extract_context_from_messages.

    These bypass the import pathway and test the extraction method directly
    against known input messages.
    """

    def test_extracts_urls(self, mgr: AgentSessionManager) -> None:
        msgs = [
            SessionMessage(role="user", content="Look at https://pypi.org/project/proxima/"),
            SessionMessage(role="assistant", content="Also see http://localhost:8080/docs"),
        ]
        ctx = mgr._extract_context_from_messages(msgs)
        urls = ctx.get("last_mentioned_urls", [])
        assert "https://pypi.org/project/proxima/" in urls
        assert "http://localhost:8080/docs" in urls

    def test_extracts_packages(self, mgr: AgentSessionManager) -> None:
        msgs = [
            SessionMessage(role="user", content="pip install requests flask>=2.0"),
        ]
        ctx = mgr._extract_context_from_messages(msgs)
        packages = ctx.get("installed_packages", [])
        assert "requests" in packages
        assert "flask" in packages

    def test_extracts_branches(self, mgr: AgentSessionManager) -> None:
        msgs = [
            SessionMessage(role="user", content="switch to develop"),
        ]
        ctx = mgr._extract_context_from_messages(msgs)
        branches = ctx.get("last_mentioned_branches", [])
        assert "develop" in branches

    def test_empty_messages_returns_valid_dict(self, mgr: AgentSessionManager) -> None:
        ctx = mgr._extract_context_from_messages([])
        assert isinstance(ctx, dict)

    def test_no_matches_still_valid(self, mgr: AgentSessionManager) -> None:
        msgs = [
            SessionMessage(role="user", content="Hello, how are you?"),
        ]
        ctx = mgr._extract_context_from_messages(msgs)
        assert isinstance(ctx, dict)
        assert ctx.get("last_mentioned_urls", []) == []

    def test_deduplication(self, mgr: AgentSessionManager) -> None:
        """Same URL mentioned twice should only appear once."""
        msgs = [
            SessionMessage(role="user", content="See https://example.com"),
            SessionMessage(role="assistant", content="Visiting https://example.com"),
        ]
        ctx = mgr._extract_context_from_messages(msgs)
        urls = ctx.get("last_mentioned_urls", [])
        assert urls.count("https://example.com") == 1


# ── 10. SessionContext Bridge Round-Trip ──────────────────────────

class TestSessionContextBridge:
    """Verify SessionState can be serialized, deserialized, and used
    to restore an AgentSessionManager session accurately."""

    def test_full_round_trip_with_tools_and_todos(self, mgr: AgentSessionManager) -> None:
        session = mgr.create_session(title="Bridge Test")
        sid = session.session_id

        # Add messages including tool results
        mgr.add_message(SessionMessage(role="user", content="Run ls"))
        mgr.add_message(SessionMessage(
            role="tool",
            content="ls output",
            tool_calls=[{"name": "ls", "arguments": {}}],
            tool_results=[{"tool_name": "ls", "output": "file1\nfile2"}],
        ))
        mgr.add_message(SessionMessage(role="assistant", content="Here are your files."))

        # Add todos
        mgr.add_todo("Task A")
        mgr.add_todo("Task B")
        mgr.update_todo_status(0, "completed")

        # Save → Load
        mgr._save_session(sid)
        del mgr._sessions[sid]
        loaded = mgr.load_session(sid)

        assert loaded.title == "Bridge Test"
        assert loaded.message_count == 3
        assert loaded.messages[1].tool_results[0]["tool_name"] == "ls"
        assert len(loaded.todos) == 2
        assert loaded.todos[0].status == "completed"
        assert loaded.todos[1].status == "pending"

    def test_session_state_to_dict_and_back(self) -> None:
        state = SessionState(
            title="Dict Bridge",
            message_count=2,
            prompt_tokens=50,
            completion_tokens=30,
            todos=[TodoItem(content="Do it", status="in_progress")],
            messages=[
                SessionMessage(role="user", content="q1"),
                SessionMessage(role="assistant", content="a1", model="test/model"),
            ],
        )
        data = state.to_dict()
        assert isinstance(data, dict)

        restored = SessionState.from_dict(data)
        assert restored.title == "Dict Bridge"
        assert restored.prompt_tokens == 50
        assert restored.completion_tokens == 30
        assert len(restored.todos) == 1
        assert restored.todos[0].status == "in_progress"
        assert restored.messages[1].model == "test/model"
