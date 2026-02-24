"""System prompt builder for the agentic loop (Phase 10, Step 10.2).

Constructs dynamic system prompts for the integrated LLM that include
current session state, available capabilities, execution instructions,
and safety rules.  The total prompt is kept under ~2 000 tokens so that
conversation history and model responses have room in the context window.
"""

from __future__ import annotations

import platform
import sys
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .robust_nl_processor import SessionContext
    from .tool_registry import ToolRegistry, RegisteredTool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

# Approximate token budget (1 token ≈ 4 chars)
_MAX_PROMPT_CHARS = 8_000  # ≈ 2 000 tokens

_ROLE_SECTION = (
    "You are Proxima's AI agent, a terminal-based assistant for quantum "
    "computing simulation. You execute tasks on the user's local machine "
    "through terminal commands. You have access to the file system, git, "
    "package managers, and build tools."
)

_EXECUTION_INSTRUCTIONS = """\
To execute a command, describe it naturally. For example:
- 'Run git clone https://github.com/user/repo'
- 'Install numpy and scipy'
- 'List files in the current directory'
- 'Read the contents of config.yaml'

For multi-step tasks, describe all steps. I will create a plan and ask \
for your confirmation before executing.

When you need to run a terminal command, state it clearly with backticks: \
`command here`

Always report results after execution. If something fails, suggest a fix."""

_SAFETY_RULES = (
    "Never execute destructive operations without user confirmation. "
    "Always create backups before modifying files. "
    "Report errors clearly with suggested fixes."
)

# Phase 16 — Todo tracking instructions for the agent
_TASK_TRACKING_INSTRUCTIONS = """\
## Task Tracking

You have a `todos` tool for managing structured task lists. Use it when:
- Starting work on a complex task (create the plan)
- Beginning each step (mark as in_progress before starting)
- Completing each step (mark as completed immediately after)
- The user asks for multiple things at once

Rules:
- Exactly ONE task in_progress at any time
- Mark tasks completed IMMEDIATELY after finishing
- Never print todo lists in your response — the user sees them in the UI
- Include both 'content' (imperative) and 'active_form' (present continuous) for each task"""


# ── Helper ────────────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars*, appending '…' if shortened."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


# ── Builder ───────────────────────────────────────────────────────────

class SystemPromptBuilder:
    """Builds the system prompt for the agentic LLM loop.

    The prompt is assembled from five sections:

    1. **Role** – static identity description.
    2. **Current State** – dynamic session state (cwd, OS, terminals …).
    3. **Capabilities** – tool registry summary, grouped by category.
    4. **Execution Instructions** – how the model should request actions.
    5. **Safety Rules** – guardrails for destructive operations.

    The builder enforces a character budget so the prompt never exceeds
    roughly 2 000 tokens (≈ 8 000 characters).
    """

    def __init__(
        self,
        tool_registry: Optional["ToolRegistry"] = None,
        max_chars: int = _MAX_PROMPT_CHARS,
    ) -> None:
        self._tool_registry = tool_registry
        self._max_chars = max_chars

    # ── public API ────────────────────────────────────────────────────

    def build(
        self,
        context: Optional["SessionContext"] = None,
        capabilities: Optional[List[str]] = None,
    ) -> str:
        """Assemble and return the full system prompt.

        Parameters
        ----------
        context:
            Current ``SessionContext`` (may be *None* at startup).
        capabilities:
            Explicit capability descriptions.  When *None*, generated
            automatically from the tool registry.
        """
        sections: List[str] = []

        # § 1 — Role (always included)
        sections.append(_ROLE_SECTION)

        # § 2 — Current State (dynamic, truncatable)
        state_section = self._build_state_section(context)
        sections.append(state_section)

        # § 3 — Capabilities
        cap_section = self._build_capabilities_section(capabilities)
        sections.append(cap_section)

        # § 4 — Execution Instructions (always included)
        sections.append(_EXECUTION_INSTRUCTIONS)

        # § 4b — Task Tracking Instructions (Phase 16)
        sections.append(_TASK_TRACKING_INSTRUCTIONS)

        # § 5 — Safety Rules (always included)
        sections.append(_SAFETY_RULES)

        prompt = "\n\n".join(s for s in sections if s)

        # Enforce budget: if too long, trim state section first
        if len(prompt) > self._max_chars:
            state_budget = max(200, self._max_chars - (len(prompt) - len(state_section)))
            state_section = _truncate(state_section, state_budget)
            sections[1] = state_section
            prompt = "\n\n".join(s for s in sections if s)

        # Final hard trim if still too long
        if len(prompt) > self._max_chars:
            prompt = _truncate(prompt, self._max_chars)

        return prompt

    # ── private helpers ───────────────────────────────────────────────

    def _build_state_section(self, context: Optional["SessionContext"]) -> str:
        """Build the current-state section from *context*."""
        lines: List[str] = ["Current state:"]

        # OS / Python
        lines.append(f"- Operating system: {platform.system()} {platform.release()}")
        lines.append(f"- Python: {sys.version.split()[0]}")

        if context is None:
            lines.append("- Working directory: (unknown)")
            return "\n".join(lines)

        # Working directory
        cwd = getattr(context, "current_directory", None) or "(unknown)"
        lines.append(f"- Working directory: {cwd}")

        # Active terminals
        terminals: Dict[str, Any] = getattr(context, "active_terminals", {})
        if terminals:
            running = [
                t.get("name", tid)
                for tid, t in terminals.items()
                if t.get("state") in ("RUNNING", "STARTING")
            ]
            lines.append(
                f"- Active terminals: {len(terminals)} "
                f"({len(running)} running: {', '.join(running[:3])})"
            )
        else:
            lines.append("- Active terminals: 0")

        # Last operation
        last_op = getattr(context, "last_operation", None)
        if last_op is not None:
            op_name = getattr(last_op, "intent_type", None)
            if op_name is not None:
                op_name = getattr(op_name, "name", str(op_name))
            lines.append(f"- Last operation: {op_name or 'none'}")
        else:
            lines.append("- Last operation: none")

        # Recent directories (max 3)
        paths: List[str] = getattr(context, "last_mentioned_paths", [])
        if paths:
            lines.append(f"- Recent directories: {', '.join(paths[:3])}")

        # Last built backend
        backend = getattr(context, "last_built_backend", None)
        if backend:
            lines.append(f"- Last built backend: {backend}")

        # Installed packages this session
        pkgs: List[str] = getattr(context, "installed_packages", [])
        if pkgs:
            lines.append(f"- Session packages: {', '.join(pkgs[:5])}")

        return "\n".join(lines)

    def _build_capabilities_section(
        self, explicit: Optional[List[str]] = None,
    ) -> str:
        """Build the capabilities section.

        Uses *explicit* descriptions when provided, otherwise reads from the
        tool registry.
        """
        if explicit:
            numbered = "\n".join(
                f"{i}. {cap}" for i, cap in enumerate(explicit, 1)
            )
            return f"You can perform these operations:\n{numbered}"

        if self._tool_registry is None:
            return self._default_capabilities()

        try:
            all_tools: List["RegisteredTool"] = self._tool_registry.get_all_tools()
        except Exception:
            return self._default_capabilities()

        if not all_tools:
            return self._default_capabilities()

        # Group by category
        by_category: Dict[str, List[str]] = {}
        for rt in all_tools:
            tool = rt.tool_instance if hasattr(rt, "tool_instance") else rt
            cat_name = "general"
            if hasattr(tool, "category"):
                cat_val = tool.category
                cat_name = getattr(cat_val, "name", str(cat_val)).lower()
            desc = getattr(tool, "description", getattr(tool, "name", "tool"))
            by_category.setdefault(cat_name, []).append(desc)

        lines = ["You can perform these operations:"]
        for cat, descs in sorted(by_category.items()):
            lines.append(f"\n**{cat.replace('_', ' ').title()}:**")
            for d in descs[:6]:  # cap per category to save space
                lines.append(f"  - {_truncate(d, 100)}")

        return "\n".join(lines)

    @staticmethod
    def _default_capabilities() -> str:
        """Fallback capability list when no registry is available."""
        return (
            "You can perform these operations:\n"
            "1. File operations — read, write, create, delete, move, search files\n"
            "2. Directory operations — list, navigate, create directories\n"
            "3. Terminal commands — run shell commands and scripts\n"
            "4. Git operations — clone, pull, push, commit, branch, merge, diff\n"
            "5. Package management — install, check, and configure dependencies\n"
            "6. Backend operations — build, test, configure, modify quantum backends\n"
            "7. Analysis — analyze execution results, export data\n"
            "8. System — check system info, manage terminals"
        )
