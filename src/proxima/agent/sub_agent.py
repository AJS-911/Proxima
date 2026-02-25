"""Sub-Agent System — Phase 16, Step 16.1: Lightweight Sub-Agent Delegation.

Enables the main agent to spawn lightweight, **read-only** sub-agents for
search, context gathering, and web-content analysis tasks.  Sub-agents
operate with a restricted tool set and cannot modify files or state.

Architecture Note
-----------------
The assistant architecture remains stable.  Any integrated model (Ollama,
API, etc.) operates dynamically through natural language understanding.
Sub-agents use the *small* model preference for cost efficiency; the main
agent continues to use the *large* model.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from proxima.agent.dynamic_tools.robust_nl_processor import SessionContext
    from proxima.agent.dynamic_tools.tool_registry import ToolRegistry
    from proxima.intelligence.llm_router import LLMRouter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

#: Default read-only tools available to every sub-agent.
SUB_AGENT_READ_ONLY_TOOLS: List[str] = [
    "read_file",
    "list_directory",
    "search_files",
    "file_info",
    "git_status",
    "git_log",
    "git_diff",
    "run_command",       # restricted to read-only commands at runtime
    "get_working_directory",
]

#: Commands that are safe for sub-agents to run via ``RunCommandTool``.
_SAFE_READ_COMMANDS: frozenset[str] = frozenset({
    "ls", "dir", "cat", "type", "head", "tail", "find", "grep",
    "wc", "file", "stat", "echo", "pwd", "whoami", "uname", "date",
    "hostname", "tree", "which", "where", "Get-ChildItem", "Get-Content",
    "Get-Location", "Select-String",
})

#: Default sub-agent system prompt.
_SUB_AGENT_SYSTEM_PROMPT = (
    "You are a research sub-agent. Your task is to find and return "
    "information.\nYou can ONLY read files, search, and browse — you "
    "CANNOT modify anything.\nBe concise and return only the relevant "
    "information requested."
)


# ═══════════════════════════════════════════════════════════════════════════
#  SubAgentConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SubAgentConfig:
    """Configuration for a sub-agent instance.

    Attributes
    ----------
    name : str
        Descriptive name (e.g. ``"Search Agent"``).
    allowed_tools : list of str
        Restricted tool set — read-only tools only.
    model_preference : str
        ``"small"`` or ``"large"`` — which model role to use.
    max_iterations : int
        Maximum agentic-loop iterations.
    auto_approve_permissions : bool
        Always ``True`` for sub-agents (read-only, safe by design).
    parent_session_id : str
        Links to the parent session for UI display.
    timeout_seconds : int
        Maximum wall-clock time.
    """

    name: str = "Research Agent"
    allowed_tools: List[str] = field(default_factory=lambda: list(SUB_AGENT_READ_ONLY_TOOLS))
    model_preference: str = "small"
    max_iterations: int = 10
    auto_approve_permissions: bool = True
    parent_session_id: str = ""
    timeout_seconds: int = 120


# ═══════════════════════════════════════════════════════════════════════════
#  SubAgent
# ═══════════════════════════════════════════════════════════════════════════

class SubAgent:
    """A lightweight, read-only sub-agent for research and context gathering.

    Sub-agents run a simplified agentic loop with a restricted tool set.
    They auto-approve all permission requests and prefer the small model
    for cost efficiency.

    Parameters
    ----------
    config : SubAgentConfig
        Sub-agent configuration.
    llm_router : LLMRouter or None
        The LLM router to use for generating responses.
    tool_registry : ToolRegistry or None
        The tool registry for looking up allowed tools.
    session_manager : optional
        Session manager for creating sub-agent sessions.
    """

    def __init__(
        self,
        config: SubAgentConfig,
        llm_router: Optional["LLMRouter"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        session_manager: Any = None,
        tool_permissions: Any = None,
    ) -> None:
        self._config = config
        self._llm_router = llm_router
        self._tool_registry = tool_registry
        self._session_manager = session_manager
        self._tool_permissions = tool_permissions

        # Filtered tool set: only expose allowed tools
        self._allowed_set = frozenset(config.allowed_tools)

        # Create a sub-agent session if session manager is available
        self._session_id: Optional[str] = None
        if session_manager is not None:
            try:
                sub_session = session_manager.create_session(
                    title=f"Sub-agent: {config.name}",
                    parent_session_id=config.parent_session_id or None,
                    is_sub_agent=True,
                )
                self._session_id = sub_session.session_id
            except Exception:
                logger.debug("Sub-agent session creation failed")

    # ── Public API ────────────────────────────────────────────────────

    def run(self, prompt: str) -> str:
        """Execute the sub-agent with the given *prompt*.

        Runs a simplified agentic loop: sends the prompt to the LLM,
        parses any tool calls, executes safe read-only tools, feeds
        results back, and repeats until a final answer is produced or
        the iteration / timeout limit is reached.

        Returns
        -------
        str
            The final response text from the sub-agent.
        """
        if self._llm_router is None:
            return self._fallback_response(prompt)

        start_time = time.time()
        conversation: List[Dict[str, str]] = [
            {"role": "system", "content": _SUB_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        collected = ""

        for iteration in range(self._config.max_iterations):
            # Timeout enforcement
            elapsed = time.time() - start_time
            if elapsed >= self._config.timeout_seconds:
                return collected or f"⏱️ Sub-agent timed out after {int(elapsed)}s."

            # Call the LLM
            response_text = self._call_llm(conversation)
            if response_text is None:
                return collected or "⚠️ Sub-agent LLM call failed."

            # Try to extract tool calls from the response
            tool_calls = self._extract_tool_calls(response_text)

            if not tool_calls:
                # No tool call — this is the final answer
                return response_text

            # Execute each tool call (only if tool is in the allowed set)
            tool_results: List[str] = []
            for tool_name, tool_args in tool_calls:
                result = self._execute_tool(tool_name, tool_args)
                tool_results.append(f"[Tool: {tool_name}]\n{result}")

            combined = "\n\n".join(tool_results)
            conversation.append({"role": "assistant", "content": response_text})
            conversation.append({"role": "user", "content": f"Tool results:\n{combined}"})
            collected = response_text

        return collected or "Sub-agent reached iteration limit."

    # ── Internal helpers ──────────────────────────────────────────────

    def _call_llm(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Send messages to the LLM and return the response text."""
        if self._llm_router is None:
            return None

        try:
            from proxima.intelligence.llm_router import LLMRequest
        except ImportError:
            return None

        # Build a single prompt from the messages
        parts: List[str] = []
        system_prompt = None
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                parts.append(f"Assistant: {msg['content']}")
        prompt = "\n\n".join(parts)

        try:
            request = LLMRequest(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=1024,
            )
            response = self._llm_router.route(request)
            if response and hasattr(response, "text"):
                return response.text or ""
            if response and isinstance(response, str):
                return response
        except Exception as exc:
            logger.debug("Sub-agent LLM call failed: %s", exc)
        return None

    def _extract_tool_calls(
        self, text: str,
    ) -> List[tuple[str, Dict[str, Any]]]:
        """Extract tool call requests from LLM response text.

        Looks for JSON blocks like ``{"tool": "name", "arguments": {...}}``.
        """
        import json
        import re

        calls: List[tuple[str, Dict[str, Any]]] = []

        # Pattern 1: fenced JSON block
        for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                if "tool" in data:
                    calls.append((data["tool"], data.get("arguments", {})))
            except (json.JSONDecodeError, KeyError):
                pass

        # Pattern 2: bare inline JSON — use a greedy approach that handles
        # one level of nested braces (e.g. {"tool": "x", "arguments": {"k": "v"}})
        if not calls:
            for match in re.finditer(
                r'(\{"tool"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\})',
                text,
                re.DOTALL,
            ):
                try:
                    data = json.loads(match.group(1))
                    if "tool" in data:
                        calls.append((data["tool"], data.get("arguments", {})))
                except (json.JSONDecodeError, KeyError):
                    pass

        return calls

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool if it's in the allowed set.

        For ``run_command``, enforces the safe-read-commands restriction.
        Respects ``auto_approve_permissions`` — when ``True`` (default for
        sub-agents) all permission checks are skipped automatically.
        """
        # Check if tool is allowed
        if tool_name not in self._allowed_set:
            return f"⚠️ Tool '{tool_name}' is not available to sub-agents."

        # For run_command, restrict to safe read-only commands
        if tool_name == "run_command":
            command = str(arguments.get("command", ""))
            base_cmd = command.strip().split()[0] if command.strip() else ""
            if base_cmd not in _SAFE_READ_COMMANDS:
                return (
                    f"⚠️ Command '{base_cmd}' is not in the sub-agent safe "
                    f"command list. Sub-agents can only use read-only commands."
                )

        # Phase 16 — Permission check (skip when auto_approve_permissions=True)
        if (
            self._tool_permissions is not None
            and not self._config.auto_approve_permissions
        ):
            try:
                action = arguments.get("command", tool_name)
                perm = self._tool_permissions.check_permission(
                    self._session_id or "", tool_name, str(action), arguments,
                )
                # Import lazily to avoid circular deps
                from proxima.agent.tool_permissions import PermissionResult
                if perm == PermissionResult.DENIED:
                    reason = self._tool_permissions.get_blocked_reason(str(action))
                    return f"🚫 Blocked: {reason or 'Denied by permission rules.'}"
                if perm == PermissionResult.NEEDS_CONSENT:
                    return f"🚫 Sub-agent cannot prompt for consent — operation skipped: {tool_name}"
            except Exception:
                pass  # If permission check fails, proceed with execution

        # Look up the tool in the registry
        if self._tool_registry is None:
            return f"⚠️ Tool registry not available."

        try:
            registered = self._tool_registry.get_tool(tool_name)
            if registered is None:
                return f"⚠️ Tool '{tool_name}' not found in registry."

            tool_instance = registered.instance
            if tool_instance is None:
                return f"⚠️ Tool '{tool_name}' has no instance."

            result = tool_instance.execute(arguments, context=None)
            if result is None:
                return "(no output)"
            if hasattr(result, "message"):
                return str(result.message or result.result or "(no output)")
            return str(result)
        except Exception as exc:
            return f"⚠️ Tool execution failed: {exc}"

    @staticmethod
    def _fallback_response(prompt: str) -> str:
        """Fallback when no LLM is available."""
        return (
            "⚠️ Sub-agent could not produce a response — no LLM router "
            "is available. The original request was:\n" + prompt[:500]
        )
