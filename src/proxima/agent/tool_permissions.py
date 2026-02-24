"""Tool Permission System — Phase 16, Step 16.3.

Provides a Crush-inspired permission layer that **wraps and extends** the
existing ``SafetyManager`` from ``safety.py``.  Adds: YOLO mode, an extended
blocked-commands list, the three-option consent flow (Allow Once / Allow for
Session / Deny), and configurable allowed-tools from YAML.

Architecture Note
-----------------
``ToolPermissionManager`` delegates blocking and base consent flow to the
existing ``SafetyManager`` — it does NOT duplicate blocked-command patterns
or the consent data model.  The assistant architecture remains stable.
"""

from __future__ import annotations

import logging
import re
import threading
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Permission result
# ═══════════════════════════════════════════════════════════════════════════

class PermissionResult(Enum):
    """Outcome of a permission check."""

    ALLOWED = "allowed"
    DENIED = "denied"                 # blocked command — cannot be overridden
    NEEDS_CONSENT = "needs_consent"   # user must approve


# ═══════════════════════════════════════════════════════════════════════════
#  Defaults
# ═══════════════════════════════════════════════════════════════════════════

#: Tools that never require permission (read-only, safe by design).
DEFAULT_ALLOWED_TOOLS: List[str] = [
    "ReadFileTool",
    "read_file",
    "ListDirectoryTool",
    "list_directory",
    "SearchFilesTool",
    "search_files",
    "FileInfoTool",
    "file_info",
    "GetWorkingDirectoryTool",
    "get_working_directory",
    "GitStatusTool",
    "git_status",
    "GitLogTool",
    "git_log",
    "GitDiffTool",
    "git_diff",
    "todos",
]

#: Extended blocked-command patterns (supplements SafetyManager.BLOCKED_PATTERNS).
EXTENDED_BLOCKED_COMMANDS: List[str] = [
    r"rm\s+-rf\s+/\*",
    r"rm\s+-rf\s+~",
    r"mkfs\.",
    r"dd\s+if=",
    r"dd\s+of=/dev/",
    r"chmod\s+-R\s+000\s+/",
    r"poweroff",
    r"init\s+0",
    r"init\s+6",
    r">\s*/dev/sda",
    r"cat\s+/dev/zero\s*>",
    r"mv\s+/\s+/dev/null",
    r"wget\s+-O-\s*\|\s*sh",
    r"curl\s.*\|\s*sh",
    r"curl\s.*\|\s*bash",
    r"python\s+-c\s+['\"]import\s+os;\s*os\.remove",
    r"del\s+/F\s+/S\s+/Q\s+C:\\",
]

#: Read-only commands that are always safe for ``RunCommandTool``.
SAFE_READ_COMMANDS: frozenset[str] = frozenset({
    "ls", "dir", "cat", "type", "head", "tail", "find", "grep",
    "wc", "file", "stat", "echo", "pwd", "whoami", "uname", "date",
    "hostname", "tree", "which", "where",
    # PowerShell equivalents
    "Get-ChildItem", "Get-Content", "Get-Location", "Select-String",
    "Get-Process", "Get-Service", "Get-Item", "Get-ItemProperty",
    "Test-Path", "Measure-Object",
})


# ═══════════════════════════════════════════════════════════════════════════
#  ToolPermissionManager
# ═══════════════════════════════════════════════════════════════════════════

class ToolPermissionManager:
    """Per-tool consent system with session-scoped auto-approval.

    Wraps the existing ``SafetyManager`` and adds:

    * A configurable list of tools that never need consent.
    * An extended blocked-command list.
    * A YOLO mode that auto-approves everything except blocked commands.
    * Session-scoped approvals (``Allow for Session``).
    * A ``check_permission()`` method that returns ``ALLOWED``, ``DENIED``,
      or ``NEEDS_CONSENT``.

    Parameters
    ----------
    safety_manager : object or None
        The existing ``SafetyManager`` instance (delegate to it for
        blocking and consent). May be *None* in minimal environments.
    config : dict
        Configuration from ``configs/default.yaml`` under ``agent.permissions``.
    """

    def __init__(
        self,
        safety_manager: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._safety = safety_manager
        cfg = config or {}

        # Allowed tools (never need consent)
        custom = cfg.get("allowed_tools", None)
        if custom and isinstance(custom, list):
            self._allowed_tools: Set[str] = set(DEFAULT_ALLOWED_TOOLS) | set(custom)
        else:
            self._allowed_tools = set(DEFAULT_ALLOWED_TOOLS)

        # Session-scoped permissions: session_id → set of "tool:action" keys
        self._session_permissions: Dict[str, Set[str]] = {}

        # YOLO mode (auto-approve everything except blocked commands)
        self._skip_all: bool = bool(cfg.get("yolo_mode", False))

        # Extended blocked-command regex patterns (compiled once)
        self._extended_blocked_re = [
            re.compile(p, re.IGNORECASE) for p in EXTENDED_BLOCKED_COMMANDS
        ]

        # Pending consent requests: request_id → threading.Event
        self._pending: Dict[str, threading.Event] = {}
        self._pending_results: Dict[str, bool] = {}
        self._lock = threading.Lock()

    # ── Properties ────────────────────────────────────────────────────

    @property
    def skip_all(self) -> bool:
        """Whether YOLO mode is active."""
        return self._skip_all

    @skip_all.setter
    def skip_all(self, value: bool) -> None:
        self._skip_all = value

    @property
    def allowed_tools(self) -> Set[str]:
        """The set of tools that never require consent."""
        return self._allowed_tools

    # ── Core API ──────────────────────────────────────────────────────

    def check_permission(
        self,
        session_id: str,
        tool_name: str,
        action: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> PermissionResult:
        """Check whether a tool execution is allowed.

        Parameters
        ----------
        session_id : str
            Current session identifier.
        tool_name : str
            Tool class name or registered name.
        action : str
            Description of the action (e.g. the command string for
            ``RunCommandTool``).
        params : dict, optional
            Tool parameters.

        Returns
        -------
        PermissionResult
            ``ALLOWED``, ``DENIED``, or ``NEEDS_CONSENT``.
        """
        params = params or {}

        # ── 1. Blocked-command check (absolute — cannot be overridden) ──
        command = str(params.get("command", action))
        if self._is_blocked(command):
            return PermissionResult.DENIED

        # ── 2. YOLO mode — approve everything except blocked ──
        if self._skip_all:
            return PermissionResult.ALLOWED

        # ── 3. Allowed-tools list ──
        if tool_name in self._allowed_tools:
            return PermissionResult.ALLOWED

        # ── 4. SafetyManager safe-operation check ──
        if self._safety is not None:
            try:
                if self._safety.is_safe_operation(tool_name):
                    return PermissionResult.ALLOWED
            except Exception:
                pass

        # ── 5. Session-scoped auto-approval ──
        key = f"{tool_name}:{action}"
        session_perms = self._session_permissions.get(session_id, set())
        if key in session_perms:
            return PermissionResult.ALLOWED

        # Also check tool-only key (without action) for broad session approval
        if tool_name in session_perms:
            return PermissionResult.ALLOWED

        # ── 6. Safe read-only command check ──
        if tool_name in ("run_command", "RunCommandTool"):
            base_cmd = command.strip().split()[0] if command.strip() else ""
            if base_cmd in SAFE_READ_COMMANDS:
                return PermissionResult.ALLOWED

        # ── 7. Needs consent ──
        return PermissionResult.NEEDS_CONSENT

    def grant_session_permission(
        self,
        session_id: str,
        tool_name: str,
        action: str = "",
    ) -> None:
        """Grant a session-scoped permission for a tool.

        Parameters
        ----------
        session_id : str
            The session to grant the permission in.
        tool_name : str
            Tool name.
        action : str
            Specific action string. If empty, grants for all actions of
            this tool.
        """
        if session_id not in self._session_permissions:
            self._session_permissions[session_id] = set()

        key = f"{tool_name}:{action}" if action else tool_name
        self._session_permissions[session_id].add(key)

    def clear_session_permissions(self, session_id: str) -> None:
        """Clear all session-scoped approvals for a session."""
        self._session_permissions.pop(session_id, None)

    def request_consent(
        self,
        session_id: str,
        tool_name: str,
        action: str,
        description: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        consent_callback: Optional[Callable] = None,
    ) -> bool:
        """Request user consent for a tool execution.

        Displays a consent dialog via the callback with three options:

        * **Allow** (once) — grants for this single execution.
        * **Allow for Session** — auto-approves future identical requests.
        * **Deny** — blocks this execution.

        If no callback is supplied but the ``SafetyManager`` has one, it
        is used as fallback.

        Parameters
        ----------
        session_id : str
            Current session.
        tool_name : str
            Tool name.
        action : str
            Action description.
        description : str
            Human-readable description for the consent dialog.
        params : dict, optional
            Tool parameters.
        consent_callback : callable, optional
            A callback ``f(description: str) -> str`` that returns one
            of ``"allow"``, ``"allow_session"``, ``"deny"``.

        Returns
        -------
        bool
            ``True`` if the user approved, ``False`` if denied.
        """
        params = params or {}

        # Build consent text
        param_lines = "\n".join(
            f"  {k}: {str(v)[:200]}" for k, v in params.items()
        )
        consent_text = (
            f"🔒 Permission Required\n\n"
            f"Tool: {tool_name}\n"
            f"Action: {action}\n"
            f"Description: {description}\n"
        )
        if param_lines:
            consent_text += f"Parameters:\n{param_lines}\n"
        consent_text += (
            "\nOptions:\n"
            "  [1] Allow (once)\n"
            "  [2] Allow for Session\n"
            "  [3] Deny\n"
        )

        # Try the provided callback first, then the safety manager's
        cb = consent_callback
        if cb is None and self._safety is not None:
            cb = getattr(self._safety, "consent_callback", None)

        if cb is not None:
            try:
                result = cb(consent_text)
                if result is None:
                    return False

                # Parse result
                result_str = str(result).lower().strip()
                if result_str in ("allow", "1", "allow_once", "approved", "yes", "y"):
                    return True
                elif result_str in ("allow_session", "2", "approved_session", "session"):
                    self.grant_session_permission(session_id, tool_name, action)
                    return True
                else:
                    return False
            except Exception:
                return False

        # If no callback, default to deny for safety
        logger.warning(
            "No consent callback available — denying '%s' for safety.", tool_name,
        )
        return False

    def get_blocked_reason(self, command: str) -> Optional[str]:
        """Return a human-readable explanation if *command* is blocked.

        Returns ``None`` if the command is not blocked.
        """
        # Check SafetyManager first
        if self._safety is not None:
            try:
                if self._safety.is_blocked(command):
                    return (
                        f"The command '{command[:80]}' matches a blocked pattern "
                        f"in Proxima's safety rules. This command is potentially "
                        f"destructive and cannot be executed, even in YOLO mode."
                    )
            except Exception:
                pass

        # Check extended patterns
        for pattern_re in self._extended_blocked_re:
            if pattern_re.search(command):
                return (
                    f"The command '{command[:80]}' matches an extended blocked "
                    f"pattern. This command is potentially destructive."
                )

        return None

    # ── Internal helpers ──────────────────────────────────────────────

    def _is_blocked(self, command: str) -> bool:
        """Check if a command is blocked by any blocklist."""
        if not command or not command.strip():
            return False

        # Delegate to SafetyManager's blocklist first
        if self._safety is not None:
            try:
                if self._safety.is_blocked(command):
                    return True
            except Exception:
                pass

        # Check extended blocked patterns
        for pattern_re in self._extended_blocked_re:
            if pattern_re.search(command):
                return True

        return False
