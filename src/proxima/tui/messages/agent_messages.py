"""Textual Message classes for inter-screen communication — Phase 9.

These messages allow the agent layer (via ``TerminalTUIBridge``) to
push events directly into the TUI widget tree, complementing the
state-based polling approach with Textual's native message system.

Usage
-----
From any component with access to ``app``::

    app.post_message(AgentTerminalStarted(
        terminal_id="term_abc12345",
        name="build backend",
        command="python setup.py build",
    ))

Screens that want to react override ``on_agent_terminal_started`` etc.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from textual.message import Message


# ═══════════════════════════════════════════════════════════════════════
# Terminal lifecycle messages — Step 9.3
# ═══════════════════════════════════════════════════════════════════════


class AgentTerminalStarted(Message):
    """Posted when the agent spawns a new terminal process.

    Attributes
    ----------
    terminal_id : str
        Unique terminal identifier (e.g. ``term_abc12345``).
    name : str
        Human-readable label for the terminal.
    command : str
        The shell command being executed.
    cwd : str
        Working directory of the terminal.
    metadata : dict
        Optional extra metadata (e.g. ``{"phase": "build"}``).
    """

    def __init__(
        self,
        terminal_id: str,
        name: str,
        command: str,
        cwd: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.terminal_id = terminal_id
        self.name = name
        self.command = command
        self.cwd = cwd
        self.metadata = metadata or {}


class AgentTerminalOutput(Message):
    """Posted for each new line of output from a terminal.

    Attributes
    ----------
    terminal_id : str
        Terminal that produced the output.
    line : str
        The output line (already stripped of trailing newlines).
    is_stderr : bool
        Whether the line came from stderr.
    """

    def __init__(
        self,
        terminal_id: str,
        line: str,
        is_stderr: bool = False,
    ) -> None:
        super().__init__()
        self.terminal_id = terminal_id
        self.line = line
        self.is_stderr = is_stderr


class AgentTerminalCompleted(Message):
    """Posted when a terminal process finishes.

    Attributes
    ----------
    terminal_id : str
        Terminal that completed.
    name : str
        Human-readable label.
    success : bool
        ``True`` if return code was 0.
    return_code : int | None
        Process exit code.
    duration : float
        Elapsed seconds.
    output_summary : str
        Last ~500 characters of output.
    """

    def __init__(
        self,
        terminal_id: str,
        name: str,
        success: bool,
        return_code: Optional[int] = None,
        duration: float = 0.0,
        output_summary: str = "",
    ) -> None:
        super().__init__()
        self.terminal_id = terminal_id
        self.name = name
        self.success = success
        self.return_code = return_code
        self.duration = duration
        self.output_summary = output_summary


# ═══════════════════════════════════════════════════════════════════════
# Result messages — Step 9.4
# ═══════════════════════════════════════════════════════════════════════


class AgentResultReady(Message):
    """Posted when a structured result is ready for the Result Tab.

    Attributes
    ----------
    result_data : dict
        Dictionary conforming to the schema expected by
        ``ResultsListView`` (keys: ``id``, ``name``, ``status``,
        ``backend``, ``duration``, ``metrics``, etc.).
    """

    def __init__(self, result_data: Dict[str, Any]) -> None:
        super().__init__()
        self.result_data = result_data


# ═══════════════════════════════════════════════════════════════════════
# Plan lifecycle messages — Phase 12 (Complex Task Execution)
# ═══════════════════════════════════════════════════════════════════════


class AgentPlanStarted(Message):
    """Posted when the agent begins executing a confirmed plan.

    Attributes
    ----------
    plan_id : str
        Unique identifier for the plan (UUID).
    title : str
        Human-readable plan title.
    steps : list[dict]
        Ordered list of step descriptors.  Each dict has at least
        ``step_id`` (int), ``intent_type`` (str), and ``description`` (str).
    """

    def __init__(
        self,
        plan_id: str,
        title: str,
        steps: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.plan_id = plan_id
        self.title = title
        self.steps = steps or []


class AgentPlanStepCompleted(Message):
    """Posted after each step of a plan finishes.

    Attributes
    ----------
    plan_id : str
        Plan this step belongs to.
    step_id : int
        0-based index of the step.
    intent_type : str
        Name of the intent type executed.
    success : bool
        Whether the step succeeded.
    message : str
        Human-readable summary of the step result.
    """

    def __init__(
        self,
        plan_id: str,
        step_id: int,
        intent_type: str,
        success: bool,
        message: str = "",
    ) -> None:
        super().__init__()
        self.plan_id = plan_id
        self.step_id = step_id
        self.intent_type = intent_type
        self.success = success
        self.message = message
