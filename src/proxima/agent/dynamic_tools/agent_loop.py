"""Agentic loop and streaming response architecture (Phase 10, Step 10.1).

Replaces the former single-shot 5-phase ``_generate_response()`` cascade
with a proper multi-turn agentic loop:

    observe → think → act → observe  (repeat until done)

The loop supports:

* **Direct dispatch** — high-confidence intents are executed immediately
  via ``IntentToolBridge``.
* **LLM-assisted execution** — when intent is ambiguous the integrated
  model reasons about what tool to call, sees the result, and decides
  the next step.
* **Streaming token display** — providers that support ``stream_send()``
  deliver tokens incrementally to the TUI.
* **Context-window management** — long conversations are auto-summarised
  to stay within the model's token budget.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from proxima.agent.dynamic_tools.robust_nl_processor import (
        Intent,
        IntentType,
        RobustNLProcessor,
        SessionContext,
    )
    from proxima.agent.dynamic_tools.intent_tool_bridge import IntentToolBridge
    from proxima.agent.dynamic_tools.tool_interface import ToolResult
    from proxima.agent.dynamic_tools.llm_integration import (
        LLMToolIntegration,
        ToolCall,
    )
    from proxima.intelligence.llm_router import (
        LLMRouter,
        LLMRequest,
        LLMResponse,
        _BaseProvider,
    )
    from proxima.agent.dynamic_tools.system_prompt_builder import SystemPromptBuilder
    from proxima.agent.dynamic_tools.tool_registry import ToolRegistry

# Phase 11 — Agent error handler for classification and retry
try:
    from proxima.agent.agent_error_handler import (
        AgentErrorHandler as _AgentErrorHandler,
        ErrorCategory as _ErrorCategory,
    )
    _ERROR_HANDLER_AVAILABLE = True
except ImportError:
    _ERROR_HANDLER_AVAILABLE = False

# Phase 15 — Agent session manager for persistence & auto-summarization
try:
    from proxima.agent.agent_session_manager import (
        AgentSessionManager as _AgentSessionManager,
        SessionMessage as _SessionMessage,
    )
    _SESSION_MANAGER_AVAILABLE = True
except ImportError:
    _SESSION_MANAGER_AVAILABLE = False

# Phase 16 — Sub-agent, tool permissions, dual-model router
try:
    from proxima.agent.tool_permissions import (
        ToolPermissionManager as _ToolPermissionManager,
        PermissionResult as _PermissionResult,
    )
    _TOOL_PERMISSIONS_AVAILABLE = True
except ImportError:
    _TOOL_PERMISSIONS_AVAILABLE = False

try:
    from proxima.agent.dual_model_router import (
        DualModelRouter as _DualModelRouter,
        ModelRole as _ModelRole,
    )
    _DUAL_MODEL_AVAILABLE = True
except ImportError:
    _DUAL_MODEL_AVAILABLE = False

try:
    from proxima.agent.sub_agent import SubAgent as _SubAgent, SubAgentConfig as _SubAgentConfig
    _SUB_AGENT_AVAILABLE = True
except ImportError:
    _SUB_AGENT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

# Maximum agentic-loop iterations per user message
_MAX_ITERATIONS = 10

# Confidence thresholds for intent dispatch
_HIGH_CONFIDENCE = 0.5
_LOW_CONFIDENCE = 0.2

# Approx chars-per-token heuristic
_CHARS_PER_TOKEN = 4

# Default context-window size (tokens) when the model doesn't report one
_DEFAULT_CONTEXT_WINDOW = 128_000

# Tool-result truncation in conversation history (chars)
_MAX_RESULT_CHARS = 500

# Context-window fill ratio that triggers summarisation
_CONTEXT_FILL_RATIO = 0.80

# Expected response buffer (tokens)
_RESPONSE_BUFFER_TOKENS = 1_000

# Characters that mark sentence boundaries (for streaming flush)
_SENTENCE_END_CHARS = frozenset(".!?\n")

# Set of provider names known to override stream_send with true streaming
_STREAMING_PROVIDERS = frozenset({
    "ollama",
    "openai",
    "anthropic",
    "google_gemini",
    "deepseek",
    "groq",
    "cohere",
    "xai",
    "lm_studio",
    "openrouter",
})

# Regex patterns for extracting tool calls from plain text (Priority 3)
_RE_BACKTICK_CMD = re.compile(
    r"(?:I'?ll run|Let me (?:run|execute)|Running|Execute|Executing)\s*:?\s*"
    r"`([^`]+)`",
    re.IGNORECASE,
)
_RE_INLINE_CMD = re.compile(
    r"(?:I'?ll run|Let me (?:run|execute)|Running|Execute|Executing)\s*:?\s*"
    r"(.+?)(?:\n|$)",
    re.IGNORECASE,
)

# Regex for extracting a JSON tool-call block from LLM text
_RE_JSON_TOOL_BLOCK = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)
_RE_BARE_JSON = re.compile(
    r'(\{"tool"\s*:.*?\})',
    re.DOTALL,
)


# ── Helper functions ──────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token count for *text* (1 token ≈ 4 English chars)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _truncate_result(text: str, max_chars: int = _MAX_RESULT_CHARS) -> str:
    """Truncate a tool result string for inclusion in conversation."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def _format_tool_result_message(result: "ToolResult") -> str:
    """Build a concise message summarising a *ToolResult* for the LLM."""
    parts: list[str] = []
    if result.success:
        parts.append(f"Tool '{result.tool_name}' succeeded.")
        data = result.result if result.result is not None else result.message
        if data:
            parts.append(f"Output: {_truncate_result(str(data))}")
    else:
        parts.append(f"Tool '{result.tool_name}' FAILED.")
        if result.error:
            parts.append(f"Error: {_truncate_result(result.error)}")
        if result.message:
            parts.append(f"Details: {_truncate_result(result.message)}")
        if result.suggestions:
            parts.append("Suggestions: " + "; ".join(result.suggestions[:3]))
    return "\n".join(parts)


# ── AgentLoop ─────────────────────────────────────────────────────────

class AgentLoop:
    """Core agentic loop replacing the legacy 5-phase pipeline.

    Parameters
    ----------
    nl_processor : RobustNLProcessor or None
        Hybrid NL intent recogniser.
    tool_bridge : IntentToolBridge or None
        Maps intents to tool execution.
    llm_router : LLMRouter or None
        Routes requests to the configured LLM provider.
    llm_tool_integration : LLMToolIntegration or None
        Parses structured tool calls from LLM responses.
    session_context : SessionContext or None
        Mutable session state.
    ui_callback : callable(str) -> None
        Sends a *complete* message to the TUI chat.
    stream_callback : callable(str) -> None or None
        Sends an *incremental token* to the TUI chat (streaming).
        May be *None* if the TUI does not support streaming.
    tool_registry : ToolRegistry or None
        Registry for listing capabilities in the system prompt.
    """

    # Phase 12: human-readable labels for plan step display
    _INTENT_DESCRIPTIONS: Dict[str, str] = {
        "GIT_CLONE": "Clone repository",
        "GIT_CHECKOUT": "Switch branch",
        "NAVIGATE_DIRECTORY": "Navigate to directory",
        "RUN_SCRIPT": "Execute script",
        "RUN_COMMAND": "Run command",
        "INSTALL_DEPENDENCY": "Install dependencies",
        "BACKEND_BUILD": "Build backend",
        "BACKEND_TEST": "Test backend",
        "BACKEND_CONFIGURE": "Configure Proxima to use backend",
        "ANALYZE_RESULTS": "Analyze results",
        "EXPORT_RESULTS": "Export results to Result Tab",
    }

    def __init__(
        self,
        nl_processor: Optional["RobustNLProcessor"] = None,
        tool_bridge: Optional["IntentToolBridge"] = None,
        llm_router: Optional["LLMRouter"] = None,
        llm_tool_integration: Optional["LLMToolIntegration"] = None,
        session_context: Optional["SessionContext"] = None,
        ui_callback: Optional[Callable[[str], None]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        plan_confirmation_callback: Optional[Callable[[str], bool]] = None,
        post_message_callback: Optional[Callable[[Any], None]] = None,
        session_manager: Optional["_AgentSessionManager"] = None,  # Phase 15
        sub_agent_factory: Optional[Callable] = None,              # Phase 16
        tool_permissions: Optional["_ToolPermissionManager"] = None,  # Phase 16
        dual_model_router: Optional["_DualModelRouter"] = None,    # Phase 16
    ) -> None:
        self._nl_processor = nl_processor
        self._tool_bridge = tool_bridge
        self._llm_router = llm_router
        self._llm_tool_integration = llm_tool_integration
        self._session_context = session_context
        self._ui_callback = ui_callback or (lambda _s: None)
        self._stream_callback = stream_callback
        self._tool_registry = tool_registry

        # Phase 12 — plan confirmation callback.
        # Signature: callback(plan_text: str) -> bool  (True = confirmed)
        self._plan_confirmation_callback = plan_confirmation_callback

        # Phase 12 — post a Textual Message to the TUI app.
        # Signature: callback(message_obj) -> None
        self._post_message_callback = post_message_callback

        # Phase 15 — session manager for persistence, import, summarization
        self._session_manager: Optional["_AgentSessionManager"] = None  # type: ignore[type-arg]
        if session_manager is not None and _SESSION_MANAGER_AVAILABLE:
            self._session_manager = session_manager

        # Phase 16 — sub-agent factory, tool permissions, dual-model router
        self._sub_agent_factory = sub_agent_factory
        self._tool_permissions: Optional["_ToolPermissionManager"] = None  # type: ignore[type-arg]
        if tool_permissions is not None and _TOOL_PERMISSIONS_AVAILABLE:
            self._tool_permissions = tool_permissions
        self._dual_model_router: Optional["_DualModelRouter"] = None  # type: ignore[type-arg]
        if dual_model_router is not None and _DUAL_MODEL_AVAILABLE:
            self._dual_model_router = dual_model_router

        # Build the system-prompt builder lazily
        self._prompt_builder: Optional["SystemPromptBuilder"] = None

        # Conversation history for the LLM (list of {role, content} dicts)
        self._conversation: List[Dict[str, str]] = []

        # Per-request retry tracker  {operation_key: count}
        self._retry_counts: Dict[str, int] = {}

        # Phase 12 — last captured script output (for ANALYZE_RESULTS chaining)
        self._last_script_output: str = ""

        # Phase 11 — error handler for classification, recovery, and retry
        self._error_handler: Optional["_AgentErrorHandler"] = None  # type: ignore[type-arg]
        if _ERROR_HANDLER_AVAILABLE:
            try:
                self._error_handler = _AgentErrorHandler()
            except Exception:
                pass

    # ── Public API ────────────────────────────────────────────────────

    def process_message(self, message: str) -> str:
        """Process a user message through the agentic loop.

        This is the main entry point, called from ``agent_ai_assistant.py``
        when the user sends a message.

        Returns
        -------
        str
            The final assistant response text (may be empty if the UI
            callback was used to stream incremental results).
        """
        self._retry_counts.clear()

        # Append to conversation
        self._conversation.append({"role": "user", "content": message})

        # Phase 15 — record user message in session manager
        self._record_session_message("user", message)

        # ── Turn 1: Intent recognition ────────────────────────────────
        intent = self._recognize_intent(message)

        if intent is not None:
            confidence = getattr(intent, "confidence", 0.0)
            intent_type_name = self._intent_name(intent)

            if confidence >= _HIGH_CONFIDENCE and intent_type_name != "UNKNOWN":
                # Direct dispatch
                result = self._direct_dispatch(intent, message)
                if result is not None:
                    return result

            # Medium confidence or complex intent → LLM-assisted if available
            if confidence >= _LOW_CONFIDENCE and self._llm_available():
                return self._llm_assisted_execution(message, intent)

            # Low confidence, LLM available
            if confidence < _LOW_CONFIDENCE and self._llm_available():
                return self._llm_assisted_execution(message, intent=None)

        # No intent at all
        if self._llm_available():
            return self._llm_assisted_execution(message, intent=None)

        return (
            "I'm not sure what you want me to do. Could you rephrase? "
            "(No LLM is integrated for advanced reasoning.)"
        )

    # ── Intent recognition ────────────────────────────────────────────

    def _recognize_intent(self, message: str) -> Optional["Intent"]:
        """Run the NL processor's 5-layer pipeline."""
        if self._nl_processor is None:
            return None
        try:
            return self._nl_processor.recognize_intent(message)
        except Exception as exc:
            logger.warning("Intent recognition failed: %s", exc)
            return None

    @staticmethod
    def _intent_name(intent: "Intent") -> str:
        it = getattr(intent, "intent_type", None)
        return getattr(it, "name", "UNKNOWN") if it is not None else "UNKNOWN"

    # ── Direct dispatch (high-confidence) ────────────────────────────

    def _direct_dispatch(
        self,
        intent: "Intent",
        message: str,
    ) -> Optional[str]:
        """Dispatch a high-confidence intent directly via the bridge.

        Returns the formatted response text, or *None* if dispatch is
        unavailable and the caller should fall through to LLM-assisted.
        """
        if self._tool_bridge is None:
            return None

        intent_name = self._intent_name(intent)

        # Multi-step handling
        if intent_name in ("MULTI_STEP", "PLAN_EXECUTION"):
            return self._dispatch_multi_step(intent)

        cwd = self._current_cwd()
        try:
            result = self._tool_bridge.dispatch(
                intent,
                cwd=cwd,
                session_context=self._session_context,
            )
        except Exception as exc:
            logger.error("Direct dispatch failed: %s", exc, exc_info=True)
            return f"❌ Operation failed: {exc}"

        response = self._format_result(result, intent_name)

        # Auto-fix/retry on fixable failure (Step 10.1 Result Evaluation)
        if not result.success:
            retry_response = self._try_auto_fix(intent, result, intent_name)
            if retry_response is not None:
                response = retry_response

        # Update context
        self._update_context_after_result(intent, result)
        self._conversation.append({"role": "assistant", "content": response})

        # Phase 15 — record assistant response and tool results
        self._record_session_message("assistant", response)
        self._record_session_message(
            "tool",
            _format_tool_result_message(result),
            tool_results=[{
                "tool_name": intent_name,
                "success": result.success,
                "output": (result.message or str(result.result or ""))[:500],
            }],
        )

        return response

    # ── Phase 12: Plan-based multi-step execution ─────────────────

    def _dispatch_multi_step(self, intent: "Intent") -> str:
        """Execute a MULTI_STEP intent via plan creation → confirmation → execution.

        Phase 12 flow:
        1. Build an ``ExecutionPlan`` from the sub-intents.
        2. Present the plan to the user and wait for confirmation.
        3. Execute each step sequentially with full context propagation.
        4. Post ``AgentPlanStarted`` / ``AgentPlanStepCompleted`` messages
           to the Execution Tab.
        5. On completion, push an ``AgentResultReady`` message to the
           Result Tab if the pipeline produced analysis output.
        """
        sub_intents: list = getattr(intent, "sub_intents", [])
        if not sub_intents:
            return "⚠️ No steps detected in the multi-step request."

        if self._tool_bridge is None:
            return "⚠️ Tool bridge unavailable for multi-step execution."

        # ── 1. Create plan ────────────────────────────────────────────
        plan_id = uuid.uuid4().hex[:12]
        plan_steps = self._create_plan_from_intents(sub_intents)

        # ── 2. Present plan to user ───────────────────────────────────
        plan_text = self._format_plan_for_display(plan_steps)
        self._ui_callback(plan_text)

        # ── 3. Wait for confirmation ──────────────────────────────────
        if self._plan_confirmation_callback is not None:
            try:
                confirmed = self._plan_confirmation_callback(plan_text)
            except Exception:
                confirmed = False  # safety-first: cancel on callback error
        else:
            # No callback registered — auto-approve (headless mode)
            confirmed = True

        if not confirmed:
            msg = "❌ Plan execution cancelled by user."
            self._conversation.append({"role": "assistant", "content": msg})
            return msg

        # ── 4. Post AgentPlanStarted to Execution Tab ─────────────────
        self._post_plan_started(plan_id, plan_steps)

        # ── 5. Execute steps sequentially with context propagation ────
        self._last_script_output = ""  # reset between runs
        cwd = self._current_cwd()
        results: list[str] = []
        all_succeeded = True
        last_result: Optional["ToolResult"] = None
        analysis_output: Optional[str] = None

        for i, (sub, step_info) in enumerate(zip(sub_intents, plan_steps)):
            sub_name = self._intent_name(sub)
            step_num = i + 1
            self._ui_callback(f"▶ Step {step_num}/{len(sub_intents)}: {step_info['description']}")

            # Inject context from prior steps into this sub-intent
            self._propagate_context_into_sub_intent(sub, cwd, last_result)

            try:
                r = self._tool_bridge.dispatch(
                    sub, cwd=cwd, session_context=self._session_context,
                )
                step_text = self._format_result(r, sub_name)
                results.append(f"**Step {step_num} ({sub_name}):** {step_text}")
                self._update_context_after_result(sub, r)

                # ── Context propagation ────────────────────────────────
                cwd = self._update_cwd_after_step(sub, r, cwd)

                # Capture script output for downstream analysis
                if sub_name in ("RUN_SCRIPT", "RUN_COMMAND"):
                    output_text = r.result if isinstance(r.result, str) else (
                        r.message or str(r.result or "")
                    )
                    self._last_script_output = output_text

                # Capture analysis output for Result Tab
                if sub_name in ("ANALYZE_RESULTS", "EXPORT_RESULTS"):
                    analysis_output = r.result if isinstance(r.result, str) else (
                        r.message or str(r.result or "")
                    )

                last_result = r

                # Post step completion to Execution Tab
                self._post_plan_step_completed(
                    plan_id, i, sub_name, r.success,
                    step_text[:200],
                )

                # Auto-fix/retry on fixable failure
                if not r.success:
                    retry_response = self._try_auto_fix(sub, r, sub_name)
                    if retry_response is not None:
                        results[-1] = f"**Step {step_num} ({sub_name}):** {retry_response}"
                        # Re-check: if the auto-fix text starts with ✅ it succeeded
                        if retry_response.startswith("✅"):
                            continue  # treat as success, move on
                    all_succeeded = False
                    break  # stop on unrecoverable failure

            except Exception as exc:
                results.append(f"**Step {step_num} ({sub_name}):** ❌ {exc}")
                self._post_plan_step_completed(
                    plan_id, i, sub_name, False, str(exc)[:200],
                )
                all_succeeded = False
                break  # stop on first failure

        # ── 6. Notify Execution Tab of plan completion ─────────────────
        # Post remaining steps as skipped so the plan exits RUNNING state
        if not all_succeeded:
            for skip_idx in range(len(results), len(sub_intents)):
                skip_name = self._intent_name(sub_intents[skip_idx])
                self._post_plan_step_completed(
                    plan_id, skip_idx, skip_name, False, "Skipped (previous step failed)",
                )

        # ── 7. Final report ───────────────────────────────────────────
        total = len(sub_intents)
        completed = len(results)
        if all_succeeded:
            status_line = f"\n✅ All {total} steps completed successfully"
        else:
            status_line = f"\n⚠️ {completed}/{total} steps completed (execution stopped on failure)"

        # Push to Result Tab if we have analysis output
        if analysis_output:
            self._push_result_to_tab(plan_id, intent, analysis_output)
            status_line += "\n📊 Results have been sent to the Result Tab (press 3 to view)"

        summary = "\n\n".join(results) + status_line
        self._conversation.append({"role": "assistant", "content": summary})

        # Phase 15 — record multi-step summary in session manager
        self._record_session_message("assistant", summary)

        return summary

    # ── Plan helpers ──────────────────────────────────────────────────

    def _create_plan_from_intents(
        self, sub_intents: list
    ) -> List[Dict[str, Any]]:
        """Build an ordered list of plan step descriptors from sub-intents.

        Each descriptor is a dict with keys:
        ``step_id``, ``intent_type``, ``description``, ``depends_on``.
        """

        steps: List[Dict[str, Any]] = []
        for i, sub in enumerate(sub_intents):
            name = self._intent_name(sub)
            raw = getattr(sub, "raw_message", "") or ""

            # Build a human-readable description
            base_desc = self._INTENT_DESCRIPTIONS.get(name, name.replace("_", " ").title())

            # Enrich with entity details
            entities = getattr(sub, "entities", []) or []
            detail_parts: list[str] = []
            for e in entities:
                etype = getattr(e, "entity_type", "")
                val = getattr(e, "value", "")
                if etype == "url" and val:
                    detail_parts.append(val)
                elif etype == "branch" and val:
                    detail_parts.append(val)
                elif etype in ("path", "destination", "dirname") and val:
                    detail_parts.append(val)
                elif etype == "filename" and val:
                    detail_parts.append(val)

            description = base_desc
            if detail_parts:
                description += ": " + ", ".join(detail_parts[:3])
            elif raw and len(raw) < 80:
                description += f" — {raw}"

            depends_on = [i - 1] if i > 0 else []

            steps.append({
                "step_id": i,
                "intent_type": name,
                "description": description,
                "depends_on": depends_on,
            })

        return steps

    def _format_plan_for_display(self, plan_steps: List[Dict[str, Any]]) -> str:
        """Format an execution plan as a readable message for the user."""
        total = len(plan_steps)
        lines = [f"📋 **Execution Plan** ({total} steps):"]
        for step in plan_steps:
            num = step["step_id"] + 1
            lines.append(f"  {num}. {step['description']}")
        lines.append("")
        lines.append("Type **yes** to proceed or **no** to cancel.")
        return "\n".join(lines)

    def _propagate_context_into_sub_intent(
        self,
        sub: "Intent",
        cwd: str,
        last_result: Optional["ToolResult"],
    ) -> None:
        """Inject context from prior steps into a sub-intent's entities.

        For example, if the previous step cloned a repo, a subsequent
        ``NAVIGATE_DIRECTORY`` step that has a relative path entity
        should resolve it against the cloned repo directory.

        This method mutates *sub* in place.
        """
        sub_name = self._intent_name(sub)
        entities = getattr(sub, "entities", None) or []

        # --- Resolve relative paths against current cwd ---------------
        if sub_name in ("NAVIGATE_DIRECTORY", "RUN_SCRIPT", "RUN_COMMAND"):
            for e in entities:
                etype = getattr(e, "entity_type", "")
                val = getattr(e, "value", "")
                if etype in ("path", "dirname", "destination") and val:
                    if not os.path.isabs(val):
                        resolved = os.path.normpath(os.path.join(cwd, val))
                        e.value = resolved

        # --- Inject last script output for analysis steps -------------
        if sub_name in ("ANALYZE_RESULTS", "EXPORT_RESULTS"):
            if self._last_script_output:
                try:
                    from proxima.agent.dynamic_tools.robust_nl_processor import (
                        ExtractedEntity,
                    )
                    # Only add if not already present
                    if not any(
                        getattr(e, "entity_type", "") == "script_output"
                        for e in entities
                    ):
                        entities.append(
                            ExtractedEntity(
                                entity_type="script_output",
                                value=self._last_script_output[:5000],
                                confidence=0.9,
                                source="context_propagation",
                            )
                        )
                except ImportError:
                    pass

        # --- Inject cloned repo path for dependent steps --------------
        if sub_name == "GIT_CHECKOUT" and self._session_context is not None:
            last_repo = getattr(self._session_context, "last_cloned_repo", None)
            if last_repo:
                # If there's no explicit cwd entity, set to last cloned repo
                has_cwd = any(
                    getattr(e, "entity_type", "") in ("path", "cwd")
                    for e in entities
                )
                if not has_cwd:
                    try:
                        from proxima.agent.dynamic_tools.robust_nl_processor import (
                            ExtractedEntity,
                        )
                        entities.append(
                            ExtractedEntity(
                                entity_type="cwd",
                                value=last_repo,
                                confidence=0.9,
                                source="context_propagation",
                            )
                        )
                    except ImportError:
                        pass

    def _update_cwd_after_step(
        self,
        sub: "Intent",
        result: "ToolResult",
        current_cwd: str,
    ) -> str:
        """Return the updated working directory after a step completes.

        Handles GIT_CLONE (cd into cloned repo), NAVIGATE_DIRECTORY
        (cd to new dir), and INSTALL_DEPENDENCY / BACKEND_BUILD
        (cwd stays the same unless the result explicitly changes it).
        """
        sub_name = self._intent_name(sub)

        if not result.success:
            return current_cwd

        # --- GIT_CLONE: cd into the cloned directory ------------------
        if sub_name == "GIT_CLONE":
            clone_dir = None
            if isinstance(result.result, dict):
                clone_dir = result.result.get("destination")
            if not clone_dir and self._session_context is not None:
                clone_dir = getattr(self._session_context, "last_cloned_repo", None)
            if clone_dir and os.path.isdir(clone_dir):
                if self._session_context is not None:
                    self._session_context.current_directory = clone_dir
                return clone_dir

        # --- NAVIGATE_DIRECTORY: use result as new cwd ----------------
        if sub_name == "NAVIGATE_DIRECTORY":
            new_dir = str(result.result) if result.result else None
            if new_dir and os.path.isdir(new_dir):
                return new_dir
            # Fallback: check session context
            if self._session_context is not None:
                ctx_dir = getattr(self._session_context, "current_directory", "")
                if ctx_dir and os.path.isdir(ctx_dir):
                    return ctx_dir

        # --- Default: pull from session context (may have been updated
        #     by the bridge handler) -----------------------------------
        return self._current_cwd()

    # ── TUI message posting helpers ───────────────────────────────────

    def _post_plan_started(
        self, plan_id: str, plan_steps: List[Dict[str, Any]]
    ) -> None:
        """Post an ``AgentPlanStarted`` message to the TUI."""
        if self._post_message_callback is None:
            return
        try:
            from proxima.tui.messages import AgentPlanStarted
            self._post_message_callback(
                AgentPlanStarted(
                    plan_id=plan_id,
                    title=f"Multi-step plan ({len(plan_steps)} steps)",
                    steps=plan_steps,
                )
            )
        except Exception:
            pass

    def _post_plan_step_completed(
        self,
        plan_id: str,
        step_id: int,
        intent_type: str,
        success: bool,
        message: str = "",
    ) -> None:
        """Post an ``AgentPlanStepCompleted`` message to the TUI."""
        if self._post_message_callback is None:
            return
        try:
            from proxima.tui.messages import AgentPlanStepCompleted
            self._post_message_callback(
                AgentPlanStepCompleted(
                    plan_id=plan_id,
                    step_id=step_id,
                    intent_type=intent_type,
                    success=success,
                    message=message,
                )
            )
        except Exception:
            pass

    def _push_result_to_tab(
        self,
        plan_id: str,
        intent: "Intent",
        analysis_output: str,
    ) -> None:
        """Push an ``AgentResultReady`` message to the Result Tab."""
        if self._post_message_callback is None:
            return
        try:
            from proxima.tui.messages import AgentResultReady
            raw = getattr(intent, "raw_message", "") or "Multi-step execution"
            self._post_message_callback(
                AgentResultReady(result_data={
                    "id": plan_id,
                    "name": raw[:80],
                    "title": raw[:80],
                    "timestamp": time.time(),
                    "status": "completed",
                    "metrics": {},
                    "output_summary": analysis_output[:2000],
                    "analysis": analysis_output,
                })
            )
        except Exception:
            pass

    def _try_auto_fix(
        self,
        intent: "Intent",
        result: "ToolResult",
        intent_name: str,
    ) -> Optional[str]:
        """Attempt auto-fix on a failed result (Phase 5 + Phase 11).

        Phase 11 enhancement:
        1. Classify the error via ``AgentErrorHandler``.
        2. Respect the category's max-retry limit (not just hard-coded 1).
        3. For *network/timeout/build* categories the handler decides
           whether an automatic retry is appropriate.
        4. Include the classified category in the UI message so the LLM
           and user both get richer context.
        5. Falls back to the Phase 5 ``ProjectDependencyManager`` path
           when no category-specific retry applies.

        Returns a new response string on successful retry, or *None* if
        no fix was applied.
        """
        error_text = result.error or result.message or ""
        if not error_text:
            return None

        # --- Phase 11: classify the error --------------------------------
        category = None
        fix_suggestion: Optional[str] = None

        if self._error_handler is not None:
            try:
                _raw_exit = getattr(result, "exit_code", None)
                _exit_code = _raw_exit if _raw_exit is not None else 1
                category, _summary, fix_suggestion = self._error_handler.classify_output(
                    error_text,
                    exit_code=_exit_code,
                )
            except Exception:
                pass

        #  Retry-gating keyed on intent+message
        op_key = f"{intent_name}:{getattr(intent, 'raw_message', '')}"
        attempt = self._retry_counts.get(op_key, 0)

        # --- Phase 11: category-aware auto-retry -------------------------
        if category is not None and self._error_handler is not None:
            should_retry, delay = self._error_handler.should_auto_retry(category, attempt)
            if should_retry:
                self._retry_counts[op_key] = attempt + 1
                cat_name = category.name if hasattr(category, "name") else str(category)
                self._ui_callback(
                    f"🔄 Retrying ({cat_name} error, attempt {attempt + 2})..."
                )
                if delay > 0:
                    time.sleep(delay)

                if self._tool_bridge is not None:
                    try:
                        retry_result = self._tool_bridge.dispatch(
                            intent,
                            cwd=self._current_cwd(),
                            session_context=self._session_context,
                        )
                        if retry_result.success:
                            return (
                                f"✅ Retry succeeded ({cat_name} error, "
                                f"attempt {attempt + 2}).\n"
                                + self._format_result(retry_result, intent_name)
                            )
                    except Exception as exc:
                        logger.warning("Category-aware retry failed: %s", exc)
                # Retry did not succeed — fall through to dep-manager path

        # --- Phase 5: dependency-fix path --------------------------------
        # Only allow a limited number of dep-manager fix attempts per operation
        if attempt >= 3:
            return None

        try:
            from proxima.agent.dependency_manager import ProjectDependencyManager
        except ImportError:
            # No dep manager — still return classification info if we have it
            if fix_suggestion:
                return f"💡 **Suggested fix:** {fix_suggestion}"
            return None

        cwd = self._current_cwd()
        try:
            mgr = ProjectDependencyManager()
            fix_cmd = mgr.detect_and_fix_errors(error_text, cwd)
        except Exception:
            if fix_suggestion:
                return f"💡 **Suggested fix:** {fix_suggestion}"
            return None

        if not fix_cmd:
            if fix_suggestion:
                return f"💡 **Suggested fix:** {fix_suggestion}"
            return None

        # Increment retry counter only when a fix is actually attempted
        self._retry_counts[op_key] = attempt + 1

        # Determine risk level
        risk = "medium"
        if category is not None and self._error_handler is not None:
            risk = self._error_handler.get_fix_risk_level(category)
        self._ui_callback(f"🔧 Auto-fix detected ({risk} risk): `{fix_cmd}` — attempting...")

        # Execute the fix via the tool bridge
        self._execute_raw_command(fix_cmd)

        # Retry original operation
        if self._tool_bridge is None:
            return None
        try:
            retry_result = self._tool_bridge.dispatch(
                intent,
                cwd=self._current_cwd(),
                session_context=self._session_context,
            )
        except Exception as exc:
            return f"🔧 Fix applied (`{fix_cmd}`), but retry still failed: {exc}"

        retry_response = self._format_result(retry_result, intent_name)
        return f"🔧 Auto-fixed with `{fix_cmd}`\n{retry_response}"

    # ── LLM-assisted execution (multi-turn loop) ─────────────────────

    def _llm_assisted_execution(
        self,
        message: str,
        intent: Optional["Intent"] = None,
    ) -> str:
        """Run the multi-turn agentic loop with the integrated LLM.

        The LLM can reason about what tools to call, see results, and
        decide next steps — up to ``_MAX_ITERATIONS``.
        """
        # Build system prompt
        system_prompt = self._build_system_prompt()

        # Prepare conversation for the LLM
        messages = self._prepare_llm_messages(system_prompt, message)

        # If we have an intent hint, prepend it
        if intent is not None:
            hint = self._build_intent_hint(intent)
            if hint:
                messages.append({"role": "system", "content": hint})

        collected_response = ""
        iteration = 0
        original_message = message  # Phase 15: preserve for re-queue on summarize

        while iteration < _MAX_ITERATIONS:
            iteration += 1

            # Phase 15 — check auto-summarization before LLM call
            self._check_and_summarize(original_message)

            # Context-window management
            messages = self._manage_context_window(messages, system_prompt)

            # Call the LLM  (stream the first response and any final answer)
            response_text, llm_response = self._call_llm(messages, stream=True)

            if response_text is None:
                # LLM call failed entirely
                if collected_response:
                    return collected_response
                return "⚠️ Failed to get a response from the LLM."

            # Phase 15 — update token counts from LLM response metadata
            self._update_session_tokens(llm_response)

            # Try to parse tool calls from the response
            tool_calls = self._extract_tool_calls(response_text, llm_response)

            if not tool_calls:
                # No tool call — this is the final answer
                collected_response = response_text
                self._conversation.append({"role": "assistant", "content": response_text})

                # Phase 15 — record final assistant response
                self._record_session_message("assistant", response_text)

                return response_text

            # Execute each tool call
            tool_results: list[str] = []
            for tc_name, tc_args in tool_calls:
                self._ui_callback(f"⚙️ Executing: {tc_name}...")
                result_text = self._execute_tool_call(tc_name, tc_args)
                tool_results.append(f"[Tool: {tc_name}]\n{result_text}")

                # Phase 15 — record each tool execution
                self._record_session_message(
                    "tool",
                    result_text,
                    tool_calls=[{"name": tc_name, "arguments": tc_args}],
                    tool_results=[{"tool_name": tc_name, "output": result_text[:500]}],
                )

            # Feed results back to the LLM
            combined_result = "\n\n".join(tool_results)
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"Tool results:\n{combined_result}"})
            self._conversation.append({"role": "assistant", "content": response_text})
            self._conversation.append({"role": "user", "content": f"Tool results:\n{_truncate_result(combined_result, 1000)}"})

            # Phase 15 — record intermediate assistant + tool-result exchange
            self._record_session_message("assistant", response_text)

        # Loop limit reached
        summary = (
            f"⏱️ Reached the maximum of {_MAX_ITERATIONS} iterations. "
            f"Here is what was accomplished so far:\n\n{collected_response or '(no intermediate results)'}"
        )
        self._conversation.append({"role": "assistant", "content": summary})

        # Phase 15 — record loop-exhaustion summary
        self._record_session_message("assistant", summary)

        return summary

    # ── LLM calling ──────────────────────────────────────────────────

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ) -> Tuple[Optional[str], Optional[Any]]:
        """Send *messages* to the LLM and return (text, raw_response).

        If *stream* is ``True`` and the provider supports streaming,
        tokens are delivered incrementally via ``self._stream_callback``.

        Phase 16: Prefers ``DualModelRouter.get_router(LARGE)`` for the
        main reasoning loop when a dual-model router is configured.
        """
        # Phase 16 — prefer dual-model router's LARGE model
        router = self._llm_router
        if self._dual_model_router is not None and _DUAL_MODEL_AVAILABLE:
            large_router = self._dual_model_router.get_router(_ModelRole.LARGE)
            if large_router is not None:
                router = large_router

        if router is None:
            return None, None

        # Build a single prompt from messages
        prompt = self._messages_to_prompt(messages)
        system_prompt = None
        if messages and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]

        try:
            from proxima.intelligence.llm_router import LLMRequest
        except ImportError:
            return None, None

        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=2048,
            stream=stream,
        )

        # Attempt streaming if requested and provider supports it
        if stream and self._stream_callback is not None:
            streamed = self._try_stream(request, router=router)
            if streamed is not None:
                return streamed

        # Non-streaming fallback
        try:
            response = router.route(request)
            return (response.text or ""), response
        except Exception as exc:
            logger.error("LLM route failed: %s", exc, exc_info=True)
            return None, None

    def _try_stream(
        self,
        request: "LLMRequest",
        router: Optional["LLMRouter"] = None,
    ) -> Optional[Tuple[str, Any]]:
        """Attempt a streaming call.  Returns ``(full_text, response)`` on
        success, or *None* if streaming is unavailable / fails.
        """
        _router = router or self._llm_router
        if _router is None:
            return None
        try:
            provider = _router._pick_provider(request)  # type: ignore[union-attr]
        except Exception:
            return None

        provider_name = getattr(provider, "name", "")
        if provider_name not in _STREAMING_PROVIDERS:
            return None

        # Verify the provider's stream_send is a real override
        from proxima.intelligence.llm_router import _BaseProvider
        if type(provider).stream_send is _BaseProvider.stream_send:
            return None

        # Resolve API key
        try:
            api_key = _router.api_keys.get_api_key(provider)  # type: ignore[union-attr]
        except Exception:
            api_key = None

        chunks: List[str] = []
        sentence_buffer: List[str] = []

        def _on_token(token: str) -> None:
            chunks.append(token)
            sentence_buffer.append(token)
            # Flush to UI on sentence boundaries to reduce re-renders
            if token and token[-1] in _SENTENCE_END_CHARS:
                flushed = "".join(sentence_buffer)
                sentence_buffer.clear()
                if self._stream_callback:
                    self._stream_callback(flushed)

        try:
            response = provider.stream_send(request, api_key, callback=_on_token)
            # Flush remaining buffer
            if sentence_buffer:
                remaining = "".join(sentence_buffer)
                if self._stream_callback:
                    self._stream_callback(remaining)
            full_text = "".join(chunks)
            return full_text, response
        except Exception as exc:
            logger.warning("Streaming failed, falling back: %s", exc)
            return None

    # ── Tool call extraction ─────────────────────────────────────────

    def _extract_tool_calls(
        self,
        response_text: str,
        llm_response: Optional[Any],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Extract tool calls from an LLM response.

        Priority order:
        1. Structured function calling (tool_calls in LLMResponse)
        2. JSON tool-call blocks in text
        3. Text-pattern fallback (backtick commands)
        """
        calls: List[Tuple[str, Dict[str, Any]]] = []

        # Priority 1: Structured tool calls from the response object
        if llm_response is not None:
            calls = self._extract_structured_calls(llm_response)
            if calls:
                return calls

        # Priority 1b: LLMToolIntegration parsing
        if self._llm_tool_integration is not None and llm_response is not None:
            try:
                parsed = self._llm_tool_integration.parse_tool_calls(llm_response)
                for tc in parsed:
                    calls.append((tc.tool_name, tc.arguments))
                if calls:
                    return calls
            except Exception:
                pass

        # Priority 2: JSON tool-call block in text
        calls = self._extract_json_tool_calls(response_text)
        if calls:
            return calls

        # Priority 3: Text pattern fallback
        calls = self._extract_text_commands(response_text)
        return calls

    def _extract_structured_calls(
        self, llm_response: Any,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Parse tool_calls / function_call from a structured LLMResponse."""
        calls: List[Tuple[str, Dict[str, Any]]] = []

        # Check tool_calls list (OpenAI multi-tool format)
        tool_calls = getattr(llm_response, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                name = getattr(tc, "name", None)
                args = getattr(tc, "arguments", {})
                if name:
                    calls.append((name, args if isinstance(args, dict) else {}))

        # Check single function_call (legacy OpenAI)
        if not calls:
            fc = getattr(llm_response, "function_call", None)
            if fc is not None:
                name = getattr(fc, "name", None)
                args = getattr(fc, "arguments", {})
                if name:
                    calls.append((name, args if isinstance(args, dict) else {}))

        return calls

    def _extract_json_tool_calls(
        self, text: str,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Extract JSON tool-call blocks from LLM text output."""
        calls: List[Tuple[str, Dict[str, Any]]] = []

        # Try fenced code block first
        for m in _RE_JSON_TOOL_BLOCK.finditer(text):
            parsed = self._parse_json_tool(m.group(1))
            if parsed:
                calls.append(parsed)

        # Try bare JSON
        if not calls:
            for m in _RE_BARE_JSON.finditer(text):
                parsed = self._parse_json_tool(m.group(1))
                if parsed:
                    calls.append(parsed)

        return calls

    @staticmethod
    def _parse_json_tool(raw: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Try to parse a JSON string as a tool call."""
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(obj, dict):
            return None
        name = obj.get("tool") or obj.get("name") or obj.get("tool_name")
        args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
        if name and isinstance(name, str):
            return (name, args if isinstance(args, dict) else {})
        return None

    def _extract_text_commands(
        self, text: str,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Extract shell commands from natural-language patterns in text.

        Detects patterns like:
        - "I'll run: `git clone ...`"
        - "Let me execute: `pip install ...`"
        - "Running `pytest`..."
        """
        calls: List[Tuple[str, Dict[str, Any]]] = []

        # Try backtick pattern first (more reliable)
        m = _RE_BACKTICK_CMD.search(text)
        if m:
            cmd = m.group(1).strip()
            if cmd and len(cmd) < 500:
                calls.append(("run_command", {"command": cmd}))
                return calls

        # Try inline pattern (less reliable — only if no backtick match)
        m = _RE_INLINE_CMD.search(text)
        if m:
            cmd = m.group(1).strip().rstrip(".")
            # Only if it looks like a real command (starts with a known prefix)
            _CMD_PREFIXES = (
                "git ", "pip ", "python ", "cd ", "ls ", "dir ", "mkdir ",
                "npm ", "conda ", "pytest ", "make ", "cmake ", "docker ",
                "cargo ", "go ", "node ", "powershell", "bash ", "sh ",
            )
            if cmd.lower().startswith(_CMD_PREFIXES) and len(cmd) < 500:
                calls.append(("run_command", {"command": cmd}))

        return calls

    # ── Tool execution bridge ────────────────────────────────────────

    def _execute_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """Execute a tool call and return a human-readable result string."""
        if self._tool_bridge is None:
            return f"(Tool bridge unavailable — cannot execute '{tool_name}')"

        # Phase 16 — Permission check before execution
        if self._tool_permissions is not None and _TOOL_PERMISSIONS_AVAILABLE:
            session_id = ""
            if self._session_manager is not None:
                session = self._session_manager.get_current_session()
                if session is not None:
                    session_id = session.session_id
            action = arguments.get("command", tool_name)
            perm = self._tool_permissions.check_permission(
                session_id, tool_name, str(action), arguments,
            )
            if perm == _PermissionResult.DENIED:
                reason = self._tool_permissions.get_blocked_reason(str(action))
                return (
                    f"🚫 Blocked: {reason or 'This command is blocked by safety rules.'}\n"
                    f"Consider using a safer alternative."
                )
            if perm == _PermissionResult.NEEDS_CONSENT:
                approved = self._tool_permissions.request_consent(
                    session_id, tool_name, str(action),
                    f"Agent wants to execute tool '{tool_name}'",
                    arguments,
                )
                if not approved:
                    return f"🚫 Operation cancelled by user: {tool_name}"

        cwd = self._current_cwd()

        # Phase 16 — Inject runtime dependencies into tool instances that
        # need them (e.g. AgenticFetchTool needs _llm_router and _tool_registry).
        if self._tool_registry is not None:
            tool_inst = self._tool_registry.get_tool_instance(tool_name)
            if tool_inst is not None:
                if self._llm_router is not None and not getattr(tool_inst, "_llm_router", None):
                    tool_inst._llm_router = self._llm_router  # type: ignore[attr-defined]
                if self._tool_registry is not None and not getattr(tool_inst, "_tool_registry", None):
                    tool_inst._tool_registry = self._tool_registry  # type: ignore[attr-defined]
                if self._dual_model_router is not None and not getattr(tool_inst, "_dual_model_router", None):
                    tool_inst._dual_model_router = self._dual_model_router  # type: ignore[attr-defined]

        # Phase 16 — Delegate research/search tasks to sub-agent
        _SUB_AGENT_TOOLS = {"web_search", "agentic_fetch"}
        if tool_name in _SUB_AGENT_TOOLS and _SUB_AGENT_AVAILABLE:
            query = arguments.get("query") or arguments.get("url") or arguments.get("prompt", "")
            if query:
                self._ui_callback(f"🔍 Delegating '{tool_name}' to sub-agent...")
                result_text = self._spawn_sub_agent(str(query), name=tool_name)
                self._record_session_message(
                    "tool",
                    result_text,
                    tool_calls=[{"name": tool_name, "arguments": arguments}],
                    tool_results=[{"tool_name": tool_name, "output": result_text[:500]}],
                )
                return result_text

        # Check if tool_name maps to an IntentType → use bridge dispatch
        intent = self._build_intent_from_tool_call(tool_name, arguments)
        if intent is not None:
            try:
                result = self._tool_bridge.dispatch(
                    intent, cwd=cwd, session_context=self._session_context,
                )
                self._update_context_after_result(intent, result)
                msg = _format_tool_result_message(result)

                # Phase 11: on failure, enrich with error classification
                if not result.success and self._error_handler is not None:
                    err_text = result.error or result.message or ""
                    try:
                        _raw_exit = getattr(result, "exit_code", None)
                        _exit_code = _raw_exit if _raw_exit is not None else 1
                        cat, summary, fix = self._error_handler.classify_output(
                            err_text, exit_code=_exit_code,
                        )
                        cat_name = cat.name if hasattr(cat, "name") else str(cat)
                        msg += f"\nError category: {cat_name}. {summary}"
                        if fix:
                            msg += f"\nSuggested fix: {fix}"
                    except Exception:
                        pass

                return msg
            except Exception as exc:
                return f"❌ Execution failed: {exc}"

        # Fallback: treat as direct run_command if tool_name is unknown
        if tool_name == "run_command":
            cmd = arguments.get("command", "")
            if cmd:
                return self._execute_raw_command(cmd)

        return f"⚠️ Unknown tool '{tool_name}'. Available tools can be listed with 'list tools'."

    def _spawn_sub_agent(self, prompt: str, name: str = "research") -> str:
        """Spawn a read-only sub-agent for research/search tasks.

        Phase 16 — the sub-agent uses ``ModelRole.SMALL`` and has
        access only to read-only tools.

        Parameters
        ----------
        prompt : str
            The research prompt for the sub-agent.
        name : str
            Human-readable label.

        Returns
        -------
        str
            The sub-agent's result text.
        """
        if not _SUB_AGENT_AVAILABLE:
            return "(Sub-agent module unavailable)"

        # Determine which LLM router to use (prefer small model)
        router = self._llm_router
        if self._dual_model_router is not None and _DUAL_MODEL_AVAILABLE:
            small_router = self._dual_model_router.get_router(_ModelRole.SMALL)
            if small_router is not None:
                router = small_router

        if router is None:
            return "(No LLM router available for sub-agent)"

        try:
            config = _SubAgentConfig(
                name=name,
                model_preference="small",
                max_iterations=5,
                timeout_seconds=60,
            )
            # Phase 16 — use the injected sub_agent_factory if available
            if self._sub_agent_factory is not None:
                sub = self._sub_agent_factory(
                    config=config,
                    llm_router=router,
                    tool_registry=self._tool_registry,
                    session_manager=self._session_manager,
                )
            else:
                sub = _SubAgent(
                    config=config,
                    llm_router=router,
                    tool_registry=self._tool_registry,
                    session_manager=self._session_manager,
                )
            result = sub.run(prompt)
            return result or "(Sub-agent returned no result)"
        except Exception as exc:
            logger.warning("Sub-agent '%s' failed: %s", name, exc)
            return f"(Sub-agent failed: {exc})"

    def _build_intent_from_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional["Intent"]:
        """Build an ``Intent`` object from an LLM-requested tool call.

        Maps well-known tool names back to ``IntentType`` values so the
        full ``IntentToolBridge`` dispatch pipeline (safety, consent,
        error recovery) is exercised.
        """
        try:
            from proxima.agent.dynamic_tools.robust_nl_processor import (
                Intent,
                IntentType,
                ExtractedEntity,
            )
        except ImportError:
            return None

        # Map tool names → IntentType
        _TOOL_TO_INTENT = {
            "run_command": IntentType.RUN_COMMAND,
            "read_file": IntentType.READ_FILE,
            "write_file": IntentType.WRITE_FILE,
            "create_file": IntentType.CREATE_FILE,
            "delete_file": IntentType.DELETE_FILE,
            "move_file": IntentType.MOVE_FILE,
            "copy_file": IntentType.COPY_FILE,
            "list_directory": IntentType.LIST_DIRECTORY,
            "create_directory": IntentType.CREATE_DIRECTORY,
            "search_files": IntentType.SEARCH_FILE,
            "change_directory": IntentType.NAVIGATE_DIRECTORY,
            "get_working_directory": IntentType.SHOW_CURRENT_DIR,
            "git_status": IntentType.GIT_STATUS,
            "git_commit": IntentType.GIT_COMMIT,
            "git_branch": IntentType.GIT_BRANCH,
            "git_log": IntentType.GIT_LOG,
            "git_diff": IntentType.GIT_DIFF,
            "git_add": IntentType.GIT_ADD,
            "file_info": IntentType.READ_FILE,
        }

        intent_type = _TOOL_TO_INTENT.get(tool_name)
        if intent_type is None:
            # If tool_name looks like an intent name directly
            try:
                intent_type = IntentType[tool_name.upper()]
            except (KeyError, AttributeError):
                return None

        # Build entities from arguments
        entities: list[ExtractedEntity] = []
        for key, val in arguments.items():
            entities.append(
                ExtractedEntity(
                    entity_type=key,
                    value=str(val),
                    confidence=0.9,
                    source="llm_tool_call",
                )
            )

        return Intent(
            intent_type=intent_type,
            entities=entities,
            confidence=0.95,
            raw_message=f"[tool:{tool_name}]",
            explanation=f"LLM requested tool '{tool_name}'",
        )

    def _execute_raw_command(self, command: str) -> str:
        """Run a shell command via the tool bridge and return output."""
        try:
            from proxima.agent.dynamic_tools.robust_nl_processor import (
                Intent,
                IntentType,
                ExtractedEntity,
            )
        except ImportError:
            return "(Cannot execute: NL processor not available)"

        intent = Intent(
            intent_type=IntentType.RUN_COMMAND,
            entities=[
                ExtractedEntity(
                    entity_type="command",
                    value=command,
                    confidence=1.0,
                    source="llm_direct",
                )
            ],
            confidence=1.0,
            raw_message=command,
            explanation=f"Run command: {command}",
        )

        if self._tool_bridge is None:
            return "(Tool bridge unavailable)"

        try:
            result = self._tool_bridge.dispatch(
                intent,
                cwd=self._current_cwd(),
                session_context=self._session_context,
            )
            self._update_context_after_result(intent, result)
            return _format_tool_result_message(result)
        except Exception as exc:
            return f"❌ Command failed: {exc}"

    # ── Context window management (Step 10.5) ────────────────────────

    def _manage_context_window(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
    ) -> List[Dict[str, str]]:
        """Trim or summarise messages if approaching the model's context limit.

        When trimming occurs the internal ``_conversation`` history is also
        compressed so that subsequent calls to ``_prepare_llm_messages`` do
        not re-expand the context beyond the budget.

        Returns the (possibly trimmed) message list.
        """
        context_window = self._get_context_window()

        # Calculate current token usage
        total_chars = sum(len(m["content"]) for m in messages)
        total_tokens = total_chars // _CHARS_PER_TOKEN
        budget = context_window - _RESPONSE_BUFFER_TOKENS

        if total_tokens < int(budget * _CONTEXT_FILL_RATIO):
            return messages  # within budget

        logger.info(
            "Context window at ~%d/%d tokens — trimming",
            total_tokens, context_window,
        )

        # Strategy: keep system prompt (idx 0) + last 5 messages,
        # summarise everything in between
        if len(messages) <= 6:
            return messages  # too few to summarise

        # Split: system + old + recent
        system_msg = messages[0] if messages[0]["role"] == "system" else None
        start_idx = 1 if system_msg else 0
        recent_count = min(5, len(messages) - start_idx)
        old_messages = messages[start_idx: -recent_count] if recent_count > 0 else messages[start_idx:]
        recent_messages = messages[-recent_count:] if recent_count > 0 else []

        # Try LLM summarisation
        summary = self._summarise_messages(old_messages)

        # Rebuild the LLM messages list
        trimmed: List[Dict[str, str]] = []
        if system_msg:
            trimmed.append(system_msg)
        trimmed.append({"role": "system", "content": f"[Conversation summary]: {summary}"})
        trimmed.extend(recent_messages)

        # ── Persist summarisation into self._conversation ─────────
        # Replace the growing conversation history with a compact form
        # so that _prepare_llm_messages does not re-expand context.
        conv_recent_count = min(5, len(self._conversation))
        conv_recent = self._conversation[-conv_recent_count:] if conv_recent_count else []
        self._conversation = [
            {"role": "system", "content": f"[Conversation summary]: {summary}"},
        ] + conv_recent
        logger.debug(
            "Conversation compressed: %d entries kept (incl. summary)",
            len(self._conversation),
        )

        return trimmed

    def _summarise_messages(
        self, messages: List[Dict[str, str]],
    ) -> str:
        """Produce a text summary of *messages*.

        Uses the LLM if available; otherwise drops to a heuristic.
        """
        if not messages:
            return "(no prior context)"

        # Heuristic: concatenate key parts
        key_parts: list[str] = []
        for m in messages:
            content = m["content"]
            role = m["role"]
            if role == "user":
                key_parts.append(f"User: {content[:120]}")
            elif role == "assistant":
                key_parts.append(f"AI: {content[:120]}")
            else:
                key_parts.append(content[:80])

        concat = "\n".join(key_parts)

        # If LLM available, ask it to summarise
        if self._llm_available():
            try:
                from proxima.intelligence.llm_router import LLMRequest
                req = LLMRequest(
                    prompt=(
                        "Summarise this conversation so far in 200 words. "
                        "Focus on actions taken, results obtained, and current state:\n\n"
                        + concat[:3000]
                    ),
                    temperature=0.2,
                    max_tokens=300,
                )
                resp = self._llm_router.route(req)  # type: ignore[union-attr]
                if resp.text:
                    return resp.text
            except Exception:
                pass

        # Fallback: just keep last few entries
        return concat[-500:]

    def _get_context_window(self) -> int:
        """Return the estimated context window for the active model."""
        if self._llm_router is None:
            return _DEFAULT_CONTEXT_WINDOW

        try:
            # Try reading from settings
            settings = getattr(self._llm_router, "settings", None)
            if settings:
                llm_cfg = getattr(settings, "llm", None)
                if llm_cfg:
                    cw = getattr(llm_cfg, "context_window", None)
                    if cw and isinstance(cw, int) and cw > 0:
                        return cw
        except Exception:
            pass

        return _DEFAULT_CONTEXT_WINDOW

    # ── System prompt ─────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Build the system prompt using ``SystemPromptBuilder``."""
        if self._prompt_builder is None:
            try:
                from proxima.agent.dynamic_tools.system_prompt_builder import (
                    SystemPromptBuilder,
                )
                self._prompt_builder = SystemPromptBuilder(
                    tool_registry=self._tool_registry,
                )
            except ImportError:
                return self._fallback_system_prompt()

        return self._prompt_builder.build(context=self._session_context)

    @staticmethod
    def _fallback_system_prompt() -> str:
        """Minimal system prompt when the builder is unavailable."""
        return (
            "You are Proxima's AI agent for quantum computing simulation. "
            "You can run terminal commands, manage files, use git, "
            "install packages, and build backends. "
            "State commands clearly with backticks."
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _llm_available(self) -> bool:
        """Return ``True`` if an LLM router is configured and usable."""
        return self._llm_router is not None

    def _current_cwd(self) -> str:
        """Return the current working directory from context or ``os.getcwd()``."""
        if self._session_context is not None:
            cwd = getattr(self._session_context, "current_directory", None)
            if cwd:
                return cwd
        return os.getcwd()

    def _prepare_llm_messages(
        self,
        system_prompt: str,
        current_message: str,
    ) -> List[Dict[str, str]]:
        """Prepare the messages array for the LLM call.

        Includes: system prompt + last 10 conversation history entries +
        current user message.
        """
        msgs: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        # Include recent conversation (up to 10 entries)
        recent = self._conversation[-10:] if len(self._conversation) > 10 else self._conversation[:]
        # The last entry is the current user message (already appended)
        # Include history entries except the last one (current msg)
        for entry in recent[:-1]:
            msgs.append(entry)

        # Current user message
        msgs.append({"role": "user", "content": current_message})

        return msgs

    def _build_intent_hint(self, intent: "Intent") -> str:
        """Build a hint message for the LLM based on partial intent recognition."""
        intent_name = self._intent_name(intent)
        confidence = getattr(intent, "confidence", 0.0)
        explanation = getattr(intent, "explanation", "")

        entities_desc = ""
        entities = getattr(intent, "entities", [])
        if entities:
            parts = []
            for e in entities[:5]:
                etype = getattr(e, "entity_type", "unknown")
                val = getattr(e, "value", "")
                parts.append(f"{etype}='{val}'")
            entities_desc = ", ".join(parts)

        return (
            f"[Intent hint: {intent_name} (confidence={confidence:.2f}). "
            f"{explanation}. Entities: {entities_desc or 'none'}. "
            f"Use this context to decide what to do.]"
        )

    def _messages_to_prompt(
        self, messages: List[Dict[str, str]],
    ) -> str:
        """Flatten messages into a single prompt string.

        Most providers can parse role-based messages, but some local
        models (Ollama with certain model formats) expect a flat prompt.
        The ``LLMRequest.system_prompt`` field handles system separately,
        so we only flatten user/assistant turns here.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                continue  # handled via system_prompt field
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(content)
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _format_result(self, result: "ToolResult", intent_name: str) -> str:
        """Format a ``ToolResult`` for display."""
        if result.success:
            msg = result.message or str(result.result or "Done.")
            return f"✅ **{intent_name}**: {msg}"
        else:
            msg = result.message or result.error or "Unknown error"
            base = f"❌ **{intent_name}**: {msg}"
            if result.suggestions:
                base += "\n💡 " + "\n💡 ".join(result.suggestions[:3])
            return base

    def _update_context_after_result(
        self,
        intent: "Intent",
        result: "ToolResult",
    ) -> None:
        """Update ``SessionContext`` with the result of an operation.

        Performs both generic updates (last_operation, conversation
        history, operation_history) and **intent-specific** updates
        (script tracking, directory stack, package installs, modified
        files, active terminals, update_from_intent) as required by
        Phase 2 Step 2.5.
        """
        if self._session_context is None:
            return

        ctx = self._session_context
        success = getattr(result, "success", False)

        # ── Generic updates ───────────────────────────────────────

        # Update last operation
        try:
            ctx.last_operation = intent
        except Exception:
            pass

        # Update last_operation_result
        try:
            res_text = result.message or str(result.result or "")
            ctx.last_operation_result = res_text[:500]
        except Exception:
            pass

        # Update context from intent entities (paths, branches, urls,
        # packages, scripts) — Phase 2 Step 2.5
        try:
            if hasattr(ctx, "update_from_intent"):
                ctx.update_from_intent(intent)
        except Exception:
            pass

        # Add to conversation history
        try:
            intent_name = self._intent_name(intent)
            raw = getattr(intent, "raw_message", "")
            if hasattr(ctx, "add_conversation_entry"):
                ctx.add_conversation_entry(raw, intent_name)
        except Exception:
            pass

        # Add to operation history
        try:
            history = getattr(ctx, "operation_history", None)
            if history is not None:
                history.append(intent)
                # Keep bounded
                if len(history) > 50:
                    del history[:10]
        except Exception:
            pass

        # ── Intent-specific updates (only on success) ─────────────
        if not success:
            return

        try:
            from proxima.agent.dynamic_tools.robust_nl_processor import IntentType
        except ImportError:
            return

        it = getattr(intent, "intent_type", None)
        if it is None:
            return

        _get = getattr(intent, "get_entity", None)
        _get_all = getattr(intent, "get_all_entities", None)

        try:
            # Script execution → track last script
            if it == IntentType.RUN_SCRIPT and _get:
                script = (
                    _get("script_path")
                    or _get("script")
                    or _get("path")
                )
                if script and hasattr(ctx, "last_script_executed"):
                    ctx.last_script_executed = script

            # Directory navigation → push directory onto stack
            if it == IntentType.NAVIGATE_DIRECTORY and _get:
                new_dir = _get("path") or _get("dirname")
                if new_dir and hasattr(ctx, "push_directory"):
                    ctx.push_directory(new_dir)

            # Package installation → record each package
            if it == IntentType.INSTALL_DEPENDENCY and _get_all:
                for pkg in _get_all("package"):
                    if hasattr(ctx, "add_package"):
                        ctx.add_package(pkg)
                    pkgs = getattr(ctx, "installed_packages", None)
                    if pkgs is not None and pkg not in pkgs:
                        pkgs.append(pkg)

            # File/backend modifications → track modified files
            if it in (
                IntentType.WRITE_FILE,
                IntentType.CREATE_FILE,
                IntentType.BACKEND_MODIFY,
            ) and _get:
                file_path = (
                    _get("path")
                    or _get("filename")
                    or _get("file")
                )
                if file_path and hasattr(ctx, "last_modified_files"):
                    ctx.last_modified_files = [file_path]

            # Backend build → track last built backend
            if it == IntentType.BACKEND_BUILD and _get:
                backend = _get("name") or _get("backend") or _get("path")
                if backend and hasattr(ctx, "last_built_backend"):
                    ctx.last_built_backend = backend

            # Git clone → record clone
            if it == IntentType.GIT_CLONE and _get:
                url = _get("url")
                if url and hasattr(ctx, "record_clone"):
                    clone_path = _get("path") or ctx.current_directory
                    ctx.record_clone(url, clone_path)

            # Environment configuration → record active environment
            if it == IntentType.CONFIGURE_ENVIRONMENT and _get:
                import os as _os
                env_name = (
                    _get("environment")
                    or _get("env_name")
                    or _get("name")
                    or ".venv"
                )
                env_path = _os.path.join(ctx.current_directory, env_name)
                active_envs = getattr(ctx, "active_environments", None)
                if active_envs is not None:
                    active_envs[env_name] = env_path

            # Terminal operations → update active_terminals dict
            if it in (
                IntentType.TERMINAL_MONITOR,
                IntentType.TERMINAL_OUTPUT,
                IntentType.TERMINAL_LIST,
            ):
                # Extract terminal IDs from result and record them
                import re as _re
                active_terms = getattr(ctx, "active_terminals", None)
                if active_terms is not None:
                    res_str = str(result.result or result.message or "")
                    for tid_match in _re.finditer(r"term_[a-f0-9]+", res_str):
                        tid = tid_match.group(0)
                        if tid not in active_terms:
                            active_terms[tid] = {
                                "state": "active",
                                "last_update": res_str[:200],
                            }

            # Terminal kill → remove from active terminals
            if it == IntentType.TERMINAL_KILL and _get:
                tid = _get("process_id") or _get("terminal")
                active_terms = getattr(ctx, "active_terminals", None)
                if tid and active_terms and tid in active_terms:
                    del active_terms[tid]

        except Exception:
            pass  # Never let context updates break the main loop

    # ── Phase 15: Session manager helpers ─────────────────────────────

    def _record_session_message(
        self,
        role: str,
        content: str,
        *,
        model: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Record a message in the session manager if available.

        This is a no-op when no session manager is configured.
        """
        if self._session_manager is None or not _SESSION_MANAGER_AVAILABLE:
            return

        try:
            msg = _SessionMessage(
                role=role,
                content=content,
                model=model,
                tool_calls=tool_calls or [],
                tool_results=tool_results or [],
            )
            self._session_manager.add_message(msg, llm_router=self._llm_router)
        except Exception as exc:
            logger.debug("Session message recording failed: %s", exc)

    def _check_and_summarize(self, original_message: str) -> None:
        """Check if auto-summarization is needed and execute it.

        If the session exceeds the context-window threshold, summarize
        older messages and notify the user.  When the agent was mid-task
        (tool calls pending), the original prompt is re-queued via a
        system-level hint so the model can resume.
        """
        if self._session_manager is None or not _SESSION_MANAGER_AVAILABLE:
            return

        try:
            context_window = self._get_context_window()
            if not self._session_manager.should_summarize(context_window):
                return

            session = self._session_manager.get_current_session()
            if session is None:
                return

            summary_msg = self._session_manager.summarize_session(
                self._llm_router, session.session_id,
            )
            if summary_msg is not None:
                self._ui_callback(
                    "📝 Session summarized to preserve context"
                )
                # Compress the internal _conversation as well so the
                # LLM prompt builder does not re-inflate the context.
                self._conversation = [
                    {"role": "system", "content": f"[Session summary]: {summary_msg.content}"},
                ] + self._conversation[-3:]

                # Step 15.3 point 5: re-queue the original task so the
                # model knows what it was working on before summarization.
                if original_message:
                    requeue_hint = (
                        "The previous session was interrupted because it got "
                        "too long. The initial user request was: "
                        f"`{original_message}`"
                    )
                    self._conversation.append(
                        {"role": "system", "content": requeue_hint}
                    )

                logger.info(
                    "Auto-summarization triggered (%d chars summary)",
                    len(summary_msg.content),
                )
        except Exception as exc:
            logger.debug("Auto-summarization check failed: %s", exc)

    def _update_session_tokens(self, llm_response: Optional[Any]) -> None:
        """Update the current session's token counters from LLM response metadata."""
        if self._session_manager is None or not _SESSION_MANAGER_AVAILABLE:
            return
        if llm_response is None:
            return

        try:
            session = self._session_manager.get_current_session()
            if session is None:
                return

            # Try to extract usage info from different response formats
            usage = getattr(llm_response, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            else:
                prompt_tokens = getattr(llm_response, "prompt_tokens", 0) or 0
                completion_tokens = getattr(llm_response, "completion_tokens", 0) or 0

            if prompt_tokens:
                session.prompt_tokens += prompt_tokens
            if completion_tokens:
                session.completion_tokens += completion_tokens
        except Exception:
            pass

    def load_session_on_startup(self) -> Optional[str]:
        """Load the most recent session on app startup (Phase 15, Step 15.3).

        Restores the ``SessionContext`` from the session and populates
        ``self._conversation`` from persisted messages.

        Returns
        -------
        str or None
            A brief status message (e.g. "Resumed session: ...") or None
            if no prior session was found.
        """
        if self._session_manager is None or not _SESSION_MANAGER_AVAILABLE:
            return None

        try:
            session = self._session_manager.load_most_recent_session()
        except Exception as exc:
            logger.debug("Failed to load most recent session: %s", exc)
            return None

        if session is None:
            return None

        # Restore conversation history from session messages
        for msg in session.messages:
            self._conversation.append({
                "role": msg.role,
                "content": msg.content,
            })

        # Restore SessionContext if available
        if (
            session.context
            and _SESSION_MANAGER_AVAILABLE
            and self._session_context is not None
        ):
            try:
                from proxima.agent.dynamic_tools.robust_nl_processor import SessionContext
                restored_ctx = SessionContext.from_dict(session.context)
                # Copy relevant fields into the existing session_context
                for attr_name in (
                    "current_directory",
                    "cloned_repositories",
                    "directory_stack",
                    "active_branches",
                    "visited_urls",
                    "installed_packages",
                    "last_cloned_repo",
                    "last_built_backend",
                    "last_script_executed",
                    "last_modified_files",
                ):
                    val = getattr(restored_ctx, attr_name, None)
                    if val is not None:
                        try:
                            setattr(self._session_context, attr_name, val)
                        except Exception:
                            pass
            except Exception as exc:
                logger.debug("SessionContext restore failed: %s", exc)

        return (
            f"Resumed session: {session.title} "
            f"({session.message_count} messages)"
        )

    def reset_conversation(self) -> None:
        """Clear the conversation history (e.g. on session reset).

        Phase 15: Also saves the current session before clearing.
        """
        # Phase 15 — save current session before clearing
        if self._session_manager is not None and _SESSION_MANAGER_AVAILABLE:
            try:
                session = self._session_manager.get_current_session()
                if session is not None:
                    self._session_manager._save_session(session.session_id)
            except Exception:
                pass

        self._conversation.clear()
        self._retry_counts.clear()
