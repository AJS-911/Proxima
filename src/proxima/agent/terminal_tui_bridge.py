"""Terminal-to-TUI Bridge — Phase 9, Steps 9.3 and 9.4.

Connects the ``TerminalOrchestrator``'s live output and completion
events to the TUI's Execution Tab (key 2) and Result Tab (key 3) via
**two complementary paths**:

1. **State-based (polling):** Output lines are appended to
   ``state.pending_execution_logs`` and results are inserted into
   ``state.experiment_results``.  The respective screens poll these
   lists on a timer.

2. **Message-based (push):** Textual ``Message`` subclasses
   (``AgentTerminalStarted``, ``AgentTerminalOutput``,
   ``AgentTerminalCompleted``, ``AgentResultReady``) are posted
   via ``app.post_message()`` so that screens can react immediately
   even when not actively polling.

Both paths are **thread-safe**: callbacks are invoked from the
orchestrator's reader threads, so writes go through
:pymod:`threading.Lock` and the Textual ``call_from_thread`` API.

Architecture Notes
~~~~~~~~~~~~~~~~~~
* No hardcoded model references.
* Works with any terminal spawned via ``TerminalOrchestrator`` regardless
  of origin (manual, LLM-generated, or agent-planned).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)

# ── optional imports ──────────────────────────────────────────────────

try:
    from proxima.agent.terminal_orchestrator import (
        TerminalOrchestrator,
        get_terminal_orchestrator,
    )
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False

try:
    from proxima.tui.messages import (
        AgentTerminalStarted,
        AgentTerminalCompleted,
        AgentTerminalOutput,
        AgentResultReady,
    )
    _MESSAGES_AVAILABLE = True
except ImportError:
    _MESSAGES_AVAILABLE = False

# ── configurable limits ───────────────────────────────────────────────

_MAX_LINE_LENGTH = 500          # truncate output lines beyond this
_MAX_PENDING_LOGS = 100         # prune state log list above this
_COMPLETION_SEPARATOR = "━" * 50


class TerminalTUIBridge:
    """Bridge between :class:`TerminalOrchestrator` events and the TUI.

    Call :meth:`attach` after the Textual ``App`` is running to start
    forwarding terminal output.

    Parameters
    ----------
    state : object
        The TUI state object (``app.state``) that has
        ``pending_execution_logs`` (list) and ``experiment_results``
        (list) attributes.
    app : textual.App, optional
        The Textual app instance — needed for ``call_from_thread``
        and ``post_message``.
    """

    def __init__(self, state: Any, app: Any = None) -> None:
        self._state = state
        self._app = app
        self._lock = threading.Lock()
        self._subscribed: Set[str] = set()
        self._orch: Optional[TerminalOrchestrator] = None

    # ── lifecycle ─────────────────────────────────────────────────

    def attach(self) -> None:
        """Attach to the global ``TerminalOrchestrator``.

        After this call, any new or existing terminal will have its
        output forwarded to the Execution Tab, and its completion
        forwarded to the Result Tab.
        """
        if not _ORCHESTRATOR_AVAILABLE:
            logger.debug("TerminalOrchestrator not available — bridge inactive")
            return
        self._orch = get_terminal_orchestrator()
        logger.info("TerminalTUIBridge attached to orchestrator")

    # ── public API ────────────────────────────────────────────────

    def subscribe_terminal(self, terminal_id: str) -> None:
        """Subscribe to a specific terminal's output stream."""
        if self._orch is None:
            return
        with self._lock:
            if terminal_id in self._subscribed:
                return
            self._subscribed.add(terminal_id)

        self._orch.subscribe_output(
            terminal_id,
            callback=lambda line: self._on_output(terminal_id, line),
        )
        logger.debug("Subscribed to terminal '%s' output", terminal_id)

    def unsubscribe_terminal(self, terminal_id: str) -> None:
        """Unsubscribe from a terminal's output stream."""
        if self._orch is None:
            return
        with self._lock:
            self._subscribed.discard(terminal_id)

    def notify_terminal_started(
        self,
        terminal_id: str,
        name: str,
        command: str,
        cwd: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Notify the TUI that a new terminal has been spawned.

        Posts an ``AgentTerminalStarted`` message and also subscribes
        to the terminal's output automatically.
        """
        self.subscribe_terminal(terminal_id)
        self._post_message_safe(
            AgentTerminalStarted,
            terminal_id=terminal_id,
            name=name,
            command=command,
            cwd=cwd,
            metadata=metadata,
        )

    def push_execution_log(
        self,
        text: str,
        level: str = "info",
        terminal_id: str = "",
    ) -> None:
        """Push a line to the Execution Tab (``pending_execution_logs``).

        Thread-safe — can be called from the orchestrator's reader
        thread.
        """
        if self._state is None:
            return

        entry: Dict[str, Any] = {
            "text": text,
            "message": text,
            "level": level,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "terminal_id": terminal_id,
        }

        with self._lock:
            logs = getattr(self._state, "pending_execution_logs", None)
            if logs is None:
                self._state.pending_execution_logs = []
                logs = self._state.pending_execution_logs
            logs.append(entry)
            # Prune old written entries to prevent unbounded growth
            if len(logs) > _MAX_PENDING_LOGS * 2:
                self._state.pending_execution_logs = [
                    e for e in logs if not e.get("written", False)
                ][-_MAX_PENDING_LOGS:]

    def push_result(self, result_data: Dict[str, Any]) -> None:
        """Push a structured result to the Result Tab (``experiment_results``).

        Thread-safe.  Also posts an ``AgentResultReady`` message.
        """
        if self._state is None:
            return

        with self._lock:
            results = getattr(self._state, "experiment_results", None)
            if results is None:
                self._state.experiment_results = []
                results = self._state.experiment_results
            # Insert at position 0 so newest results appear first
            results.insert(0, result_data)

        # Also push via Message for screens that handle it
        self._post_message_safe(AgentResultReady, result_data=result_data)

    def on_terminal_completed(self, terminal_id: str) -> None:
        """Called when a terminal finishes — pushes result to Result Tab.

        Typically called from the terminal intent handler after
        ``wait_for_completion`` returns or from an event callback.
        """
        if self._orch is None:
            return

        parsed = self._orch.parse_terminal_result(terminal_id)
        if "error" in parsed and not parsed.get("raw_output"):
            return

        status = parsed.get("status", "unknown")
        success = status == "success"
        duration = parsed.get("duration", 0)
        name = parsed.get("name", "Terminal Result")

        # Map parsed result to the format expected by ResultsListView
        result_data = {
            "id": parsed.get("id", terminal_id),
            "name": name,
            "status": status,
            "backend": "Terminal",
            "duration": duration,
            "success_rate": 1.0 if success else 0.0,
            "avg_fidelity": parsed.get("metrics", {}).get("fidelity", 0.0),
            "total_shots": parsed.get("metrics", {}).get("tests_passed", 0),
            "exit_code": parsed.get("return_code"),
            "command": parsed.get("command", ""),
            "output": parsed.get("output_summary", ""),
            "metrics": parsed.get("metrics", {}),
            "metadata": {
                "terminal_id": terminal_id,
                "cwd": parsed.get("cwd", ""),
            },
            "timestamp": parsed.get("timestamp", ""),
        }

        # Push completion log to Execution Tab
        status_icon = "✅" if success else "❌"
        self.push_execution_log(
            f"\n{_COMPLETION_SEPARATOR}\n"
            f"{status_icon} Terminal '{name}' completed "
            f"(exit code {result_data['exit_code']})\n"
            f"Duration: {duration:.2f}s",
            level="success" if success else "error",
            terminal_id=terminal_id,
        )

        # Push to Result Tab (also posts AgentResultReady)
        self.push_result(result_data)

        # Post completion message for Execution Tab
        self._post_message_safe(
            AgentTerminalCompleted,
            terminal_id=terminal_id,
            name=name,
            success=success,
            return_code=parsed.get("return_code"),
            duration=duration,
            output_summary=parsed.get("output_summary", ""),
        )

    # ── internal ──────────────────────────────────────────────────

    def _on_output(self, terminal_id: str, line: str) -> None:
        """Callback for each output line from a subscribed terminal.

        Truncates long lines to prevent TUI log overflow.
        """
        display_line = line[:_MAX_LINE_LENGTH] if len(line) > _MAX_LINE_LENGTH else line
        self.push_execution_log(
            f"[{terminal_id[-8:]}] {display_line}",
            level="info",
            terminal_id=terminal_id,
        )
        # Also push via Message for immediate rendering
        self._post_message_safe(
            AgentTerminalOutput,
            terminal_id=terminal_id,
            line=display_line,
        )

    def _post_message_safe(self, msg_cls: type, **kwargs: Any) -> None:
        """Post a Textual message thread-safely, failing silently.

        Uses ``app.call_from_thread`` when called from a non-main
        thread, otherwise uses ``app.post_message`` directly.
        """
        if not _MESSAGES_AVAILABLE or self._app is None:
            return
        try:
            msg = msg_cls(**kwargs)
            # call_from_thread is safe from any thread
            self._app.call_from_thread(self._app.post_message, msg)
        except Exception:
            # Swallow — message delivery is best-effort
            pass


# ═══════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════

_bridge: Optional[TerminalTUIBridge] = None


def get_terminal_tui_bridge(
    state: Any = None,
    app: Any = None,
) -> Optional[TerminalTUIBridge]:
    """Return the global ``TerminalTUIBridge`` singleton.

    On first call, *state* and *app* should be provided to initialise
    the bridge.  Subsequent calls may omit them.
    """
    global _bridge
    if _bridge is None and state is not None:
        _bridge = TerminalTUIBridge(state=state, app=app)
        _bridge.attach()
    return _bridge
