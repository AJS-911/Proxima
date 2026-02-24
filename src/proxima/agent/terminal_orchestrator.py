"""Terminal Orchestrator for Proxima Agent — Phase 9, Step 9.1.

Unified interface for spawning, monitoring, and controlling multiple
terminal sessions.  Wraps the existing ``MultiTerminalMonitor`` (from
``multi_terminal.py``) and ``TerminalStateMachine`` (from
``terminal_state_machine.py``) to give the agent a single entry-point
for all terminal lifecycle operations.

Features
--------
* Spawn terminal processes with automatic event tracking
* Circular output buffer (10k lines) per terminal
* Live output subscriptions for streaming to the TUI
* Graceful kill with SIGTERM → SIGKILL escalation
* Wait-for-completion with configurable timeout
* Cross-platform: PowerShell on Windows, ``$SHELL`` / ``sh`` on Unix

Architecture Notes
~~~~~~~~~~~~~~~~~~
* **No hardcoded model references.**  Any integrated model (local or
  remote) can use the orchestrator through the ``IntentToolBridge``
  dispatch layer.
* Terminals are tracked simultaneously in both the ``MultiTerminalMonitor``
  registry (for output buffering and event emission) and the
  ``TerminalStateMachine`` (for validated lifecycle transitions).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Pre-compiled regex patterns for metric extraction (Step 9.4) ──────

_RE_TIME_KV = re.compile(
    r"(?:Time|Duration|Elapsed)\s*[:=]\s*([\d.]+)\s*s", re.IGNORECASE,
)
_RE_TIME_SUFFIX = re.compile(
    r"(\d+\.\d+)\s*(?:seconds|sec|s)\s+(?:elapsed|total)", re.IGNORECASE,
)
_RE_TESTS_PASSED = re.compile(r"(\d+)\s+passed", re.IGNORECASE)
_RE_TESTS_FAILED = re.compile(r"(\d+)\s+failed", re.IGNORECASE)
_RE_TESTS_FRACTION = re.compile(
    r"Tests?\s+passed\s*[:=]\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE,
)
_RE_FIDELITY = re.compile(r"Fidelity\s*[:=]\s*([\d.]+)", re.IGNORECASE)
_RE_ACCURACY = re.compile(r"Accuracy\s*[:=]\s*([\d.]+)", re.IGNORECASE)
_RE_ENTROPY = re.compile(r"Entropy\s*[:=]\s*([\d.]+)", re.IGNORECASE)
_RE_ERROR_LINE = re.compile(
    r"(?:^|\n).*(?:error|ERROR|Error|FAIL|FATAL).*(?:\n|$)",
)
_RE_WARNING_LINE = re.compile(
    r"(?:^|\n).*(?:warning|WARNING|Warning).*(?:\n|$)",
)

# ── Imports from existing infrastructure ──────────────────────────────

try:
    from proxima.agent.multi_terminal import (
        CircularOutputBuffer,
        MultiTerminalMonitor,
        TerminalEvent,
        TerminalEventType,
        TerminalInfo,
        TerminalState,
        get_multi_terminal_monitor,
    )
    _MULTI_TERMINAL_AVAILABLE = True
except ImportError:
    _MULTI_TERMINAL_AVAILABLE = False

try:
    from proxima.agent.terminal_state_machine import (
        TerminalProcessState,
        TerminalStateMachine,
        get_terminal_state_machine,
    )
    _STATE_MACHINE_AVAILABLE = True
except ImportError:
    _STATE_MACHINE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# Internal tracking dataclass
# ═══════════════════════════════════════════════════════════════════════

class _ManagedTerminal:
    """Internal record for a terminal managed by the orchestrator."""

    __slots__ = (
        "terminal_id", "name", "command", "cwd", "background",
        "process", "output_buffer", "start_time", "end_time",
        "state", "pid", "return_code", "reader_thread",
        "output_subscribers", "metadata",
    )

    def __init__(
        self,
        terminal_id: str,
        name: str,
        command: str,
        cwd: str,
        background: bool = False,
    ):
        self.terminal_id = terminal_id
        self.name = name
        self.command = command
        self.cwd = cwd
        self.background = background
        self.process: Optional[subprocess.Popen] = None
        self.output_buffer = CircularOutputBuffer(max_lines=10_000) if _MULTI_TERMINAL_AVAILABLE else None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.state: str = "PENDING"
        self.pid: Optional[int] = None
        self.return_code: Optional[int] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.output_subscribers: List[Callable[[str], None]] = []
        self.metadata: Dict[str, Any] = {}


# ═══════════════════════════════════════════════════════════════════════
# TerminalOrchestrator
# ═══════════════════════════════════════════════════════════════════════

class TerminalOrchestrator:
    """Unified terminal-session management for the Proxima agent.

    Delegates to :class:`MultiTerminalMonitor` for event-based output
    tracking and to :class:`TerminalStateMachine` for validated
    lifecycle transitions, while providing a simpler, agent-friendly
    API.

    Usage::

        orch = TerminalOrchestrator()
        tid = orch.spawn_terminal("build", "python setup.py build", "/project")
        ok, output = orch.wait_for_completion(tid, timeout=120)
        orch.kill_terminal(tid)
    """

    # ── class-level defaults ──────────────────────────────────────

    _RECENT_COMPLETED_TTL: float = 600.0  # 10 minutes

    def __init__(self) -> None:
        self._terminals: Dict[str, _ManagedTerminal] = {}
        self._lock = threading.Lock()

        # Obtain singletons from the existing infrastructure
        self._monitor: Optional[MultiTerminalMonitor] = (
            get_multi_terminal_monitor() if _MULTI_TERMINAL_AVAILABLE else None
        )
        self._state_machine: Optional[TerminalStateMachine] = (
            get_terminal_state_machine() if _STATE_MACHINE_AVAILABLE else None
        )

    # ── public API ────────────────────────────────────────────────

    def spawn_terminal(
        self,
        name: str,
        command: str,
        cwd: str,
        background: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Spawn a new terminal process.

        Parameters
        ----------
        name : str
            Human-readable label for this terminal (e.g. "build",
            "install deps", "run tests").
        command : str
            The shell command to execute.
        cwd : str
            Working directory for the process.
        background : bool
            If *True* the caller is not expected to wait for completion.
        metadata : dict, optional
            Arbitrary metadata (e.g. ``{"phase": "build"}``).

        Returns
        -------
        str
            A unique terminal ID (``term_<hex8>``).
        """
        terminal_id = f"term_{uuid.uuid4().hex[:8]}"

        managed = _ManagedTerminal(
            terminal_id=terminal_id,
            name=name,
            command=command,
            cwd=cwd,
            background=background,
        )
        managed.metadata = metadata or {}

        # Ensure working directory exists
        effective_cwd = cwd if os.path.isdir(cwd) else os.getcwd()

        # Build the platform-appropriate shell command
        shell_cmd = self._build_shell_command(command)

        try:
            proc = subprocess.Popen(
                shell_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                shell=True,
                cwd=effective_cwd,
                text=True,
                bufsize=1,  # line-buffered
            )
        except Exception as exc:
            logger.error("Failed to spawn terminal '%s': %s", name, exc)
            managed.state = "FAILED"
            managed.end_time = time.time()
            with self._lock:
                self._terminals[terminal_id] = managed
            return terminal_id

        managed.process = proc
        managed.pid = proc.pid
        managed.start_time = time.time()
        managed.state = "RUNNING"

        # Register with MultiTerminalMonitor
        if self._monitor is not None:
            try:
                self._monitor.register(
                    command=command,
                    working_dir=effective_cwd,
                    terminal_id=terminal_id,
                    metadata={"name": name, **(metadata or {})},
                )
                self._monitor.update_state(
                    terminal_id, TerminalState.RUNNING, pid=proc.pid,
                )
            except Exception:
                logger.debug("MultiTerminalMonitor registration skipped")

        # Register with TerminalStateMachine
        if self._state_machine is not None:
            try:
                self._state_machine.create_process(terminal_id, command)
                self._state_machine.transition_sync(
                    terminal_id, TerminalProcessState.STARTING,
                )
                self._state_machine.transition_sync(
                    terminal_id, TerminalProcessState.RUNNING,
                )
            except Exception:
                logger.debug("TerminalStateMachine registration skipped")

        # Emit STARTED event
        if self._monitor is not None:
            try:
                self._monitor._emit_event(TerminalEvent(
                    event_type=TerminalEventType.STARTED,
                    terminal_id=terminal_id,
                    data={"name": name, "command": command, "pid": proc.pid},
                ))
            except Exception:
                pass

        # Start background reader thread
        reader = threading.Thread(
            target=self._read_output,
            args=(managed,),
            daemon=True,
            name=f"TermReader-{terminal_id}",
        )
        managed.reader_thread = reader
        reader.start()

        with self._lock:
            self._terminals[terminal_id] = managed

        logger.info(
            "Spawned terminal '%s' (id=%s, pid=%s): %s",
            name, terminal_id, proc.pid, command[:80],
        )
        return terminal_id

    def get_output(self, terminal_id: str, tail_lines: int = 50) -> str:
        """Return the last *tail_lines* of output for a terminal.

        Parameters
        ----------
        terminal_id : str
            Terminal identifier returned by :meth:`spawn_terminal`.
        tail_lines : int
            Number of lines to return (default 50).  Pass ``0`` to
            retrieve **all** captured lines (the falsy value bypasses
            the truncation guard).

        Returns
        -------
        str
            The combined output text, or an error message if the terminal
            does not exist.
        """
        with self._lock:
            managed = self._terminals.get(terminal_id)
        if managed is None:
            return f"[error] Terminal '{terminal_id}' not found."

        if managed.output_buffer is None:
            return "[error] Output buffer unavailable."

        lines = managed.output_buffer.get_lines()
        if tail_lines and tail_lines < len(lines):
            lines = lines[-tail_lines:]
        return "\n".join(line.content for line in lines)

    def get_all_terminals(self) -> List[Dict[str, Any]]:
        """Return info for all active and recently completed terminals.

        Completed terminals are included if they finished within the
        last 10 minutes (``_RECENT_COMPLETED_TTL``).  Also triggers
        a lightweight cleanup of entries older than 1 hour.
        """
        # Opportunistic cleanup of very old entries
        self.cleanup_stale(max_age=3600.0)

        now = time.time()
        result: List[Dict[str, Any]] = []

        with self._lock:
            terminals = list(self._terminals.values())

        for m in terminals:
            # Skip stale completed terminals
            if m.state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
                if m.end_time and (now - m.end_time) > self._RECENT_COMPLETED_TTL:
                    continue

            elapsed = (now - m.start_time) if m.start_time else 0.0
            line_count = m.output_buffer.line_count if m.output_buffer else 0

            last_5: List[str] = []
            if m.output_buffer:
                tail = m.output_buffer.get_lines()
                last_5 = [ln.content for ln in tail[-5:]]

            result.append({
                "id": m.terminal_id,
                "name": m.name,
                "command": m.command,
                "state": m.state,
                "pid": m.pid,
                "start_time": m.start_time,
                "elapsed_seconds": round(elapsed, 1),
                "output_lines_count": line_count,
                "last_5_lines": last_5,
                "return_code": m.return_code,
                "cwd": m.cwd,
                "background": m.background,
                "metadata": m.metadata,
            })

        # Sort: running first, then by start_time descending
        state_order = {"RUNNING": 0, "STARTING": 1, "PENDING": 2}
        result.sort(key=lambda d: (
            state_order.get(d["state"], 9),
            -(d["start_time"] or 0),
        ))
        return result

    def kill_terminal(self, terminal_id: str) -> bool:
        """Kill a terminal process.

        Sends ``SIGTERM`` (or ``terminate()`` on Windows), waits 5 s,
        then ``SIGKILL`` (or ``kill()``) if still alive.

        Returns *True* on success.
        """
        with self._lock:
            managed = self._terminals.get(terminal_id)
        if managed is None:
            return False
        if managed.process is None:
            return False

        # If already in a terminal state, nothing to kill
        if managed.state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
            return True

        proc = managed.process
        try:
            proc.terminate()
        except Exception:
            pass

        # Give the process up to 5 seconds to exit
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=3)
            except Exception:
                pass

        # Update state
        managed.state = "CANCELLED"
        managed.end_time = time.time()
        managed.return_code = proc.returncode

        # Update monitor
        if self._monitor is not None:
            try:
                self._monitor.update_state(
                    terminal_id, TerminalState.CANCELLED,
                    return_code=proc.returncode,
                )
            except Exception:
                pass

        # Update state machine
        if self._state_machine is not None:
            try:
                self._state_machine.transition_sync(
                    terminal_id, TerminalProcessState.CANCELLED,
                )
            except Exception:
                pass

        # Emit CANCELLED event
        if self._monitor is not None:
            try:
                self._monitor._emit_event(TerminalEvent(
                    event_type=TerminalEventType.CANCELLED,
                    terminal_id=terminal_id,
                    data={"return_code": proc.returncode},
                ))
            except Exception:
                pass

        logger.info("Killed terminal '%s' (id=%s)", managed.name, terminal_id)
        return True

    def wait_for_completion(
        self,
        terminal_id: str,
        timeout: int = 600,
    ) -> Tuple[bool, str]:
        """Block until the terminal completes or timeout is reached.

        Parameters
        ----------
        terminal_id : str
            Terminal identifier.
        timeout : int
            Maximum seconds to wait (default 600 = 10 min).

        Returns
        -------
        tuple[bool, str]
            ``(success, output_text)``  where *success* is ``True`` iff
            the process exited with return-code 0.
        """
        with self._lock:
            managed = self._terminals.get(terminal_id)
        if managed is None:
            return False, f"Terminal '{terminal_id}' not found."
        if managed.process is None:
            return False, "Process not started."

        deadline = time.time() + timeout
        while time.time() < deadline:
            poll = managed.process.poll()
            if poll is not None:
                # Process has finished
                output = self.get_output(terminal_id, tail_lines=0)
                success = (poll == 0)
                return success, output
            time.sleep(0.5)

        # Timeout — kill
        logger.warning(
            "Terminal '%s' timed out after %ds — killing.",
            managed.name, timeout,
        )
        self.kill_terminal(terminal_id)
        output = self.get_output(terminal_id, tail_lines=0)
        return False, f"[timeout] Timed out after {timeout}s.\n{output}"

    def subscribe_output(
        self,
        terminal_id: str,
        callback: Callable[[str], None],
    ) -> bool:
        """Register a callback invoked for each new output line.

        The callback is called from the reader thread, so it should be
        lightweight and thread-safe (e.g. ``app.call_from_thread``).

        Returns *True* if the subscription was registered.
        """
        with self._lock:
            managed = self._terminals.get(terminal_id)
        if managed is None:
            return False
        managed.output_subscribers.append(callback)
        return True

    def unsubscribe_output(
        self,
        terminal_id: str,
        callback: Callable[[str], None],
    ) -> bool:
        """Remove an output subscription."""
        with self._lock:
            managed = self._terminals.get(terminal_id)
        if managed is None:
            return False
        try:
            managed.output_subscribers.remove(callback)
            return True
        except ValueError:
            return False

    def get_terminal_info(self, terminal_id: str) -> Optional[Dict[str, Any]]:
        """Return info dict for a single terminal, or *None*."""
        for info in self.get_all_terminals():
            if info["id"] == terminal_id:
                return info
        return None

    def cleanup_stale(self, max_age: float = 3600.0) -> int:
        """Remove completed/failed terminals older than *max_age* seconds.

        Returns the number of terminals removed.
        """
        now = time.time()
        to_remove: List[str] = []
        with self._lock:
            for tid, m in self._terminals.items():
                if m.state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
                    if m.end_time and (now - m.end_time) > max_age:
                        to_remove.append(tid)
            for tid in to_remove:
                del self._terminals[tid]
        return len(to_remove)

    def find_terminal(
        self,
        hint: str,
    ) -> Optional[str]:
        """Resolve a fuzzy user reference to a terminal ID.

        Matches by:
        1. Exact terminal ID
        2. Ordinal ("terminal 1" / "first terminal" / "1")
        3. Name or command substring

        Returns the terminal ID or *None*.
        """
        hint_lower = hint.strip().lower()
        all_terms = self.get_all_terminals()

        if not all_terms:
            return None

        # 1. Exact ID match
        for t in all_terms:
            if t["id"] == hint_lower or t["id"] == hint.strip():
                return t["id"]

        # 2. Ordinal match
        ordinal_map = {
            "1": 0, "first": 0, "terminal 1": 0,
            "2": 1, "second": 1, "terminal 2": 1,
            "3": 2, "third": 2, "terminal 3": 2,
            "4": 3, "fourth": 3, "terminal 4": 3,
            "5": 4, "fifth": 4, "terminal 5": 4,
            "last": -1, "latest": -1, "most recent": -1, "recent": -1,
        }
        idx = ordinal_map.get(hint_lower)
        if idx is not None:
            try:
                return all_terms[idx]["id"]
            except IndexError:
                return all_terms[-1]["id"] if all_terms else None

        # Digit-only → ordinal
        m = re.match(r"^(\d+)$", hint_lower)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(all_terms):
                return all_terms[idx]["id"]

        # 3. Name / command substring
        for t in all_terms:
            if hint_lower in t["name"].lower() or hint_lower in t["command"].lower():
                return t["id"]

        return None

    # ── result parsing utilities (Step 9.4) ───────────────────────

    def parse_terminal_result(self, terminal_id: str) -> Dict[str, Any]:
        """Parse the output of a completed terminal into a structured result.

        Extracts key-value metrics from common output formats like
        ``"Time: 12.5s"``, ``"Fidelity: 0.995"``, JSON output, etc.

        Returns a dictionary suitable for the Result Tab.
        """
        with self._lock:
            managed = self._terminals.get(terminal_id)
        if managed is None:
            return {"error": f"Terminal '{terminal_id}' not found."}

        full_output = self.get_output(terminal_id, tail_lines=0)
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

        result: Dict[str, Any] = {
            "id": terminal_id,
            "name": managed.name or managed.command[:50],
            "title": managed.name or managed.command[:50],
            "timestamp": now_iso,
            "status": "success" if managed.return_code == 0 else "failed",
            "return_code": managed.return_code,
            "output_summary": full_output[:500],
            "raw_output": full_output,
            "metrics": {},
            "command": managed.command,
            "cwd": managed.cwd,
            "duration": round(
                (managed.end_time or time.time()) - (managed.start_time or time.time()),
                2,
            ),
        }

        # ── extract metrics from output (using pre-compiled patterns) ─
        metrics = result["metrics"]

        # Duration / time patterns
        for compiled_re, key in [
            (_RE_TIME_KV, "execution_time"),
            (_RE_TIME_SUFFIX, "execution_time"),
        ]:
            m = compiled_re.search(full_output)
            if m:
                metrics[key] = float(m.group(1))

        # Test results
        for compiled_re, keys in [
            (_RE_TESTS_PASSED, ("tests_passed",)),
            (_RE_TESTS_FAILED, ("tests_failed",)),
            (_RE_TESTS_FRACTION, ("tests_passed", "tests_total")),
        ]:
            m = compiled_re.search(full_output)
            if m:
                for i, k in enumerate(keys):
                    metrics[k] = int(m.group(i + 1))

        # Quantum-specific metrics
        for compiled_re, key in [
            (_RE_FIDELITY, "fidelity"),
            (_RE_ACCURACY, "accuracy"),
            (_RE_ENTROPY, "entropy"),
        ]:
            m = compiled_re.search(full_output)
            if m:
                metrics[key] = float(m.group(1))

        # Error / warning counts
        error_count = len(_RE_ERROR_LINE.findall(full_output))
        warning_count = len(_RE_WARNING_LINE.findall(full_output))
        if error_count:
            metrics["error_lines"] = error_count
        if warning_count:
            metrics["warning_lines"] = warning_count

        # Try JSON extraction (entire output is JSON)
        if full_output.strip().startswith("{"):
            try:
                parsed_json = json.loads(full_output.strip())
                if isinstance(parsed_json, dict):
                    metrics.update(
                        {k: v for k, v in parsed_json.items()
                         if isinstance(v, (int, float, str, bool))}
                    )
            except (json.JSONDecodeError, ValueError):
                pass

        return result

    # ── LLM-based result analysis (Step 9.5) ─────────────────────

    def analyze_result(
        self,
        terminal_id: str,
        llm_provider: Optional[Any] = None,
    ) -> str:
        """Analyse the output of a completed terminal.

        If *llm_provider* is given (any object exposing
        ``generate(prompt: str) -> str``), it is used for intelligent
        analysis.  Otherwise, a rule-based summary is produced.

        Parameters
        ----------
        terminal_id : str
            Terminal to analyse.
        llm_provider : object, optional
            An object with a ``generate(prompt) -> str`` method (model-
            agnostic — works with any LLM router adapter).

        Returns
        -------
        str
            Formatted analysis text.
        """
        parsed = self.parse_terminal_result(terminal_id)
        if "error" in parsed and not parsed.get("raw_output"):
            return f"❌ {parsed['error']}"

        raw = parsed.get("raw_output", "")
        metrics = parsed.get("metrics", {})
        status = parsed.get("status", "unknown")
        name = parsed.get("name", "terminal")

        # ── Try LLM analysis ─────────────────────────────────────
        if llm_provider is not None:
            try:
                prompt = (
                    "Analyze the following execution output and provide:\n"
                    "1. Summary: What was executed and what happened\n"
                    "2. Status: Success or failure with specific details\n"
                    "3. Key Metrics: Extract any numerical results "
                    "(timing, accuracy, counts)\n"
                    "4. Issues: Any warnings or errors detected\n"
                    "5. Recommendations: Next steps or improvements\n"
                    "\n"
                    f"Command: {parsed.get('command', 'N/A')}\n"
                    f"Return code: {parsed.get('return_code', 'N/A')}\n"
                    "\nOutput to analyze:\n"
                    f"{raw[:4000]}"
                )
                llm_response = llm_provider.generate(prompt)
                if llm_response and len(llm_response.strip()) > 20:
                    return (
                        f"## 🔬 Analysis: {name}\n\n"
                        f"{llm_response.strip()}"
                    )
            except Exception as exc:
                logger.debug("LLM analysis failed: %s — falling back to rule-based", exc)

        # ── Rule-based fallback ───────────────────────────────────
        parts: List[str] = []
        parts.append(f"## 📊 Analysis: {name}\n")

        # Status
        icon = "✅" if status == "success" else "❌"
        parts.append(f"**Status:** {icon} {status.upper()}")
        if parsed.get("return_code") is not None:
            parts.append(f"  (exit code {parsed['return_code']})")
        parts.append("")

        # Duration
        dur = parsed.get("duration")
        if dur:
            parts.append(f"**Duration:** {dur:.1f}s")

        # Metrics table
        if metrics:
            parts.append("\n**Key Metrics:**")
            for k, v in metrics.items():
                label = k.replace("_", " ").title()
                parts.append(f"  • {label}: {v}")

        # Error summary
        err_count = metrics.get("error_lines", 0)
        warn_count = metrics.get("warning_lines", 0)
        if err_count or warn_count:
            parts.append(f"\n**Issues:** {err_count} error(s), {warn_count} warning(s)")

        # Output snippet
        lines = raw.strip().splitlines()
        if lines:
            snippet = lines[-min(10, len(lines)):]
            parts.append("\n**Last output lines:**\n```")
            parts.extend(snippet)
            parts.append("```")

        return "\n".join(parts)

    # ── internal helpers ──────────────────────────────────────────

    @staticmethod
    def _build_shell_command(command: str) -> str:
        """Wrap *command* for the current platform's shell.

        On Windows, PowerShell is invoked with ``-EncodedCommand`` to
        avoid double-quote escaping issues.  When the encoded form is
        too long (> 8 000 chars), falls back to single-quote wrapping
        with inner-quote escaping.
        """
        if os.name == "nt":
            import base64
            try:
                encoded = base64.b64encode(
                    command.encode("utf-16-le")
                ).decode("ascii")
                if len(encoded) < 8000:
                    return (
                        f"powershell -NoProfile -EncodedCommand {encoded}"
                    )
            except Exception:
                pass
            # Fallback: escape inner single quotes
            escaped = command.replace("'", "''")
            return f"powershell -NoProfile -Command '{escaped}'"
        return command

    def _read_output(self, managed: _ManagedTerminal) -> None:
        """Background thread: read stdout and stderr line-by-line.

        Lines are stored in the circular buffer, forwarded to the
        ``MultiTerminalMonitor``, and pushed to any live subscribers.
        """
        proc = managed.process
        if proc is None:
            return

        def _read_stream(stream, is_stderr: bool) -> None:
            try:
                for raw_line in iter(stream.readline, ""):
                    if raw_line is None:
                        break
                    line = raw_line.rstrip("\n\r")

                    # Store in circular buffer
                    if managed.output_buffer is not None:
                        managed.output_buffer.append(line, is_stderr=is_stderr)

                    # Forward to MultiTerminalMonitor
                    if self._monitor is not None:
                        try:
                            self._monitor.append_output(
                                managed.terminal_id, line, is_stderr=is_stderr,
                            )
                        except Exception:
                            pass

                    # Forward to TerminalStateMachine metrics
                    if self._state_machine is not None:
                        try:
                            self._state_machine.record_output(
                                managed.terminal_id, line, is_stderr=is_stderr,
                            )
                        except Exception:
                            pass

                    # Notify subscribers
                    for cb in list(managed.output_subscribers):
                        try:
                            cb(line)
                        except Exception:
                            pass
            except Exception:
                pass

        # Read stdout and stderr in parallel threads
        stdout_thread = threading.Thread(
            target=_read_stream,
            args=(proc.stdout, False),
            daemon=True,
            name=f"StdoutReader-{managed.terminal_id}",
        )
        stderr_thread = threading.Thread(
            target=_read_stream,
            args=(proc.stderr, True),
            daemon=True,
            name=f"StderrReader-{managed.terminal_id}",
        )
        stdout_thread.start()
        stderr_thread.start()

        # Wait for both streams to finish
        stdout_thread.join()
        stderr_thread.join()

        # Wait for process exit
        try:
            proc.wait(timeout=10)
        except Exception:
            pass

        # Record completion
        managed.return_code = proc.returncode
        managed.end_time = time.time()

        if proc.returncode == 0:
            managed.state = "COMPLETED"
            terminal_state = TerminalState.COMPLETED if _MULTI_TERMINAL_AVAILABLE else None
            process_state = TerminalProcessState.COMPLETED if _STATE_MACHINE_AVAILABLE else None
        else:
            managed.state = "FAILED"
            terminal_state = TerminalState.FAILED if _MULTI_TERMINAL_AVAILABLE else None
            process_state = TerminalProcessState.FAILED if _STATE_MACHINE_AVAILABLE else None

        # Update monitor
        if self._monitor is not None and terminal_state is not None:
            try:
                self._monitor.update_state(
                    managed.terminal_id,
                    terminal_state,
                    return_code=proc.returncode,
                )
            except Exception:
                pass

        # Update state machine
        if self._state_machine is not None and process_state is not None:
            try:
                self._state_machine.transition_sync(
                    managed.terminal_id, process_state,
                )
            except Exception:
                pass

        # Emit completion event
        if self._monitor is not None:
            try:
                # Use COMPLETED for success and FAILED (if available)
                # or STATE_CHANGED as fallback for non-zero exit
                if proc.returncode == 0:
                    evt_type = TerminalEventType.COMPLETED
                else:
                    evt_type = getattr(
                        TerminalEventType, "FAILED",
                        TerminalEventType.STATE_CHANGED,
                    )
                self._monitor._emit_event(TerminalEvent(
                    event_type=evt_type,
                    terminal_id=managed.terminal_id,
                    data={
                        "return_code": proc.returncode,
                        "duration_s": round(
                            (managed.end_time or 0) - (managed.start_time or 0), 2,
                        ),
                    },
                ))
            except Exception:
                pass

        logger.info(
            "Terminal '%s' (id=%s) finished with code %s",
            managed.name, managed.terminal_id, proc.returncode,
        )


# ═══════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════

_orchestrator: Optional[TerminalOrchestrator] = None


def get_terminal_orchestrator() -> TerminalOrchestrator:
    """Return the global ``TerminalOrchestrator`` singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TerminalOrchestrator()
    return _orchestrator
