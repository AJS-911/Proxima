"""Todos Tool — Phase 16, Step 16.4.

Provides a structured task-tracking facility that the agent can use
during complex multi-step tasks.  Registered via ``@register_tool``
so it appears in the LLM's tool list automatically.

Architecture Note
-----------------
The tool stores state in ``AgentSessionManager``'s ``SessionState.todos``
list so that todo items persist across session save/restore cycles.  The
TUI renders a separate ``TodoPillWidget`` based on the metadata returned
by this tool — the agent does NOT need to print the list in its chat
response.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from proxima.agent.dynamic_tools.tool_interface import (
    BaseTool,
    ParameterType,
    PermissionLevel,
    RiskLevel,
    ToolCategory,
    ToolParameter,
    ToolResult,
)
from proxima.agent.dynamic_tools.tool_registry import register_tool

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Valid statuses
# ═══════════════════════════════════════════════════════════════════════════

_VALID_STATUSES = frozenset({"pending", "in_progress", "completed"})


# ═══════════════════════════════════════════════════════════════════════════
#  TodosTool
# ═══════════════════════════════════════════════════════════════════════════

@register_tool
class TodosTool(BaseTool):
    """Structured task-tracking tool for the agentic loop.

    The agent calls this tool to create, update, and track progress on
    complex multi-step coding tasks.  The TUI renders the todo list in a
    dedicated pill widget, so the agent should never print the list in
    its response.
    """

    # ── Metadata ──────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "todos"

    @property
    def description(self) -> str:
        return (
            "Creates and manages a structured task list for tracking "
            "progress on complex, multi-step coding tasks.\n"
            "Use this tool proactively for:\n"
            "- Complex multi-step tasks requiring 3+ distinct steps\n"
            "- After receiving new instructions to capture requirements\n"
            "- When starting work on a task (mark as in_progress BEFORE beginning)\n"
            "- After completing a task (mark completed immediately)\n"
            "Do NOT use for single, trivial tasks."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SYSTEM

    @property
    def required_permission(self) -> PermissionLevel:
        return PermissionLevel.READ_ONLY  # no consent needed — just tracking

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.NONE

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="todos",
                param_type=ParameterType.ARRAY,
                description=(
                    "The complete updated todo list.  Each item is a JSON "
                    "object with 'content' (imperative string), 'status' "
                    "('pending'|'in_progress'|'completed'), and 'active_form' "
                    "(present continuous string)."
                ),
                required=True,
                inference_hint=(
                    "Array of {content, status, active_form} objects.  "
                    "At most one item may be in_progress at a time."
                ),
            ),
        ]

    # ── Execution ─────────────────────────────────────────────────────

    def _execute(
        self,
        parameters: Dict[str, Any],
        context: Any = None,
    ) -> ToolResult:
        """Validate and persist the todo list.

        Parameters
        ----------
        parameters : dict
            Tool parameters (must contain ``todos`` key).
        context : ExecutionContext or None
            Execution context (unused by this tool).

        Returns
        -------
        ToolResult
            Success result with updated summary and metadata for the TUI.
        """
        raw_todos = parameters.get("todos")
        if raw_todos is None:
            return ToolResult.error_result(self.name, "Missing 'todos' parameter.")

        # ── Parse ─────────────────────────────────────────────────────
        items: List[Dict[str, Any]] = []
        if isinstance(raw_todos, str):
            try:
                raw_todos = json.loads(raw_todos)
            except json.JSONDecodeError:
                return ToolResult.error_result(
                    self.name,
                    "Failed to parse 'todos' as JSON array.",
                )

        if not isinstance(raw_todos, list):
            return ToolResult.error_result(
                self.name,
                "'todos' must be a JSON array of {content, status, active_form} objects.",
            )

        # ── Validate each item ────────────────────────────────────────
        in_progress_count = 0
        for idx, item in enumerate(raw_todos):
            if not isinstance(item, dict):
                return ToolResult.error_result(
                    self.name,
                    f"Item {idx} is not a JSON object (got {type(item).__name__}).",
                )

            content = item.get("content", "").strip()
            status = item.get("status", "pending").strip().lower()
            active_form = item.get("active_form", "").strip()

            if not content:
                return ToolResult.error_result(
                    self.name,
                    f"Item {idx} is missing 'content'.",
                )

            if status not in _VALID_STATUSES:
                return ToolResult.error_result(
                    self.name,
                    f"Item {idx} has invalid status '{status}'. "
                    f"Must be one of: {', '.join(sorted(_VALID_STATUSES))}.",
                )

            if status == "in_progress":
                in_progress_count += 1
                if in_progress_count > 1:
                    return ToolResult.error_result(
                        self.name,
                        "At most one task may be 'in_progress' at a time.",
                    )

            items.append({
                "content": content,
                "status": status,
                "active_form": active_form,
            })

        # ── Diff against session's existing todos ─────────────────────
        old_todos: List[Dict[str, Any]] = []
        session_manager = self._get_session_manager()
        if session_manager is not None:
            try:
                existing = session_manager.get_todos()
                old_todos = [
                    {"content": t.content, "status": t.status, "active_form": t.active_form}
                    for t in existing
                ]
            except Exception:
                pass

        is_new = len(old_todos) == 0
        old_status_map = {t["content"]: t["status"] for t in old_todos}

        just_completed: List[str] = []
        just_started: Optional[str] = None
        for item in items:
            old_stat = old_status_map.get(item["content"])
            if item["status"] == "completed" and old_stat and old_stat != "completed":
                just_completed.append(item["content"])
            if item["status"] == "in_progress" and old_stat != "in_progress":
                just_started = item["content"]

        # ── Persist ───────────────────────────────────────────────────
        if session_manager is not None:
            try:
                # Clear existing todos and re-add
                session = session_manager.get_current_session()
                if session is not None:
                    session.todos.clear()
                    for item in items:
                        session_manager.add_todo(
                            content=item["content"],
                            active_form=item.get("active_form", ""),
                        )
                        # Set status after adding (add_todo defaults to "pending")
                        if item["status"] != "pending":
                            idx = len(session.todos) - 1
                            session_manager.update_todo_status(idx, item["status"])
            except Exception as exc:
                logger.warning("Failed to persist todos: %s", exc)

        # ── Build summary ─────────────────────────────────────────────
        pending_count = sum(1 for i in items if i["status"] == "pending")
        in_prog_count = sum(1 for i in items if i["status"] == "in_progress")
        completed_count = sum(1 for i in items if i["status"] == "completed")

        message = (
            f"Todo list updated successfully.\n"
            f"Status: {pending_count} pending, {in_prog_count} in progress, "
            f"{completed_count} completed"
        )

        metadata: Dict[str, Any] = {
            "is_new": is_new,
            "todos": items,
            "just_completed": just_completed,
            "just_started": just_started,
            "completed": completed_count,
            "total": len(items),
        }

        return ToolResult.success_result(
            tool_name=self.name,
            result=message,
            message=message,
            metadata=metadata,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_session_manager(self) -> Any:
        """Attempt to locate the session manager from the execution context.

        Returns *None* if unavailable (minimal environments, tests, etc.).
        """
        # Check if set as attribute (injected during integration)
        if hasattr(self, "_session_manager") and self._session_manager is not None:
            return self._session_manager

        # Try the execution context
        ctx = getattr(self, "_execution_context", None)
        if ctx is not None:
            mgr = getattr(ctx, "session_manager", None)
            if mgr is not None:
                return mgr

        return None
