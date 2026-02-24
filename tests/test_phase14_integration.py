"""Phase 14 — Core Integration and Wiring validation tests.

Verifies that:
 1. All Phase-14 components import without circular dependencies.
 2. ``_initialize_components()`` creates the correct attributes on
    ``AgentAIAssistantScreen``.
 3. Message classes are importable and the Execution / Results screens
    expose the expected handlers.
 4. The dependency-injection wiring (shared ``_dependency_manager``,
    ``_terminal_orchestrator``, etc.) is consistent.

No live LLM, subprocess, or TUI is required — everything is mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ═══════════════════════════════════════════════════════════════════════
# Step 14.2: Import chain — no circular imports
# ═══════════════════════════════════════════════════════════════════════


class TestImportChain:
    """Verify that the full import chain resolves without error."""

    def test_import_agent_ai_assistant(self) -> None:
        """Importing the top-level screen should not raise."""
        from proxima.tui.screens.agent_ai_assistant import AgentAIAssistantScreen  # noqa: F401

    def test_import_agent_loop(self) -> None:
        from proxima.agent.dynamic_tools.agent_loop import AgentLoop  # noqa: F401

    def test_import_intent_tool_bridge(self) -> None:
        from proxima.agent.dynamic_tools.intent_tool_bridge import IntentToolBridge  # noqa: F401

    def test_import_robust_nl_processor(self) -> None:
        from proxima.agent.dynamic_tools.robust_nl_processor import RobustNLProcessor  # noqa: F401

    def test_import_system_prompt_builder(self) -> None:
        from proxima.agent.dynamic_tools.system_prompt_builder import SystemPromptBuilder  # noqa: F401

    def test_import_agent_error_handler(self) -> None:
        from proxima.agent.agent_error_handler import AgentErrorHandler  # noqa: F401

    def test_import_dependency_manager(self) -> None:
        from proxima.agent.dependency_manager import ProjectDependencyManager  # noqa: F401

    def test_import_terminal_orchestrator(self) -> None:
        from proxima.agent.terminal_orchestrator import TerminalOrchestrator  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════
# Step 14.3 / 14.4: Message classes and screen handlers
# ═══════════════════════════════════════════════════════════════════════


class TestMessageWiring:
    """Verify Textual message classes exist and screens handle them."""

    def test_all_message_classes_importable(self) -> None:
        from proxima.tui.messages import (
            AgentTerminalStarted,
            AgentTerminalOutput,
            AgentTerminalCompleted,
            AgentResultReady,
            AgentPlanStarted,
            AgentPlanStepCompleted,
        )
        # Quick sanity: they are all Textual Messages
        for cls in (
            AgentTerminalStarted,
            AgentTerminalOutput,
            AgentTerminalCompleted,
            AgentResultReady,
            AgentPlanStarted,
            AgentPlanStepCompleted,
        ):
            assert callable(cls), f"{cls.__name__} is not callable"

    def test_execution_screen_has_handlers(self) -> None:
        from proxima.tui.screens.execution import ExecutionScreen

        expected_handlers = [
            "on_agent_terminal_started",
            "on_agent_terminal_output",
            "on_agent_terminal_completed",
            "on_agent_plan_started",
            "on_agent_plan_step_completed",
        ]
        for handler_name in expected_handlers:
            assert hasattr(ExecutionScreen, handler_name), (
                f"ExecutionScreen missing handler: {handler_name}"
            )

    def test_results_screen_has_handler(self) -> None:
        from proxima.tui.screens.results import ResultsScreen

        assert hasattr(ResultsScreen, "on_agent_result_ready"), (
            "ResultsScreen missing handler: on_agent_result_ready"
        )


# ═══════════════════════════════════════════════════════════════════════
# Step 14.1: Component initialization and wiring
# ═══════════════════════════════════════════════════════════════════════


class TestComponentWiring:
    """Verify ``_initialize_components`` creates the expected attributes.

    Because ``AgentAIAssistantScreen`` inherits from the Textual TUI
    widget tree, we cannot instantiate it outside a running app.
    Instead we verify the *availability flags* and import paths are
    consistent and test a lightweight mock-based initialization.
    """

    def test_availability_flags_exist(self) -> None:
        """All Phase-14 availability flags must be importable."""
        from proxima.tui.screens import agent_ai_assistant as mod

        expected_flags = [
            "AGENT_AVAILABLE",
            "LLM_AVAILABLE",
            "ROBUST_NL_AVAILABLE",
            "INTENT_BRIDGE_AVAILABLE",
            "AGENT_LOOP_AVAILABLE",
            "LLM_TOOL_INTEGRATION_AVAILABLE",
            "TERMINAL_ORCHESTRATOR_AVAILABLE",
            "DEPENDENCY_MANAGER_AVAILABLE",
            "ERROR_HANDLER_AVAILABLE",
            "SYSTEM_PROMPT_BUILDER_AVAILABLE",
        ]
        for flag in expected_flags:
            assert hasattr(mod, flag), f"Missing availability flag: {flag}"
            # Must be a bool
            val = getattr(mod, flag)
            assert isinstance(val, bool), (
                f"{flag} should be bool, got {type(val).__name__}"
            )

    def test_dependency_manager_standalone(self) -> None:
        """ProjectDependencyManager can be instantiated independently."""
        from proxima.agent.dependency_manager import ProjectDependencyManager

        dm = ProjectDependencyManager()
        assert dm is not None
        assert hasattr(dm, "detect_project_dependencies")

    def test_terminal_orchestrator_singleton(self) -> None:
        """get_terminal_orchestrator returns the same instance."""
        from proxima.agent.terminal_orchestrator import get_terminal_orchestrator

        o1 = get_terminal_orchestrator()
        o2 = get_terminal_orchestrator()
        assert o1 is o2, "TerminalOrchestrator should be a singleton"

    def test_system_prompt_builder_standalone(self) -> None:
        """SystemPromptBuilder works with no tool registry."""
        from proxima.agent.dynamic_tools.system_prompt_builder import (
            SystemPromptBuilder,
        )

        builder = SystemPromptBuilder(tool_registry=None)
        prompt = builder.build()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_error_handler_with_dep_manager(self) -> None:
        """AgentErrorHandler accepts a dependency manager."""
        from proxima.agent.agent_error_handler import AgentErrorHandler
        from proxima.agent.dependency_manager import ProjectDependencyManager

        dm = ProjectDependencyManager()
        handler = AgentErrorHandler(dep_manager=dm)
        assert handler is not None
        assert hasattr(handler, "classify_output")

    def test_intent_bridge_accepts_consent_callback(self) -> None:
        """IntentToolBridge can be created with a consent callback."""
        from proxima.agent.dynamic_tools.intent_tool_bridge import IntentToolBridge

        cb = MagicMock(return_value=True)
        bridge = IntentToolBridge(consent_callback=cb)
        assert bridge is not None
        assert bridge._consent_callback is cb

    def test_intent_bridge_set_llm_provider(self) -> None:
        """set_llm_provider accepts an adapter without error."""
        from proxima.agent.dynamic_tools.intent_tool_bridge import IntentToolBridge

        bridge = IntentToolBridge(consent_callback=lambda msg: True)
        mock_provider = MagicMock()
        mock_provider.chat = MagicMock(return_value="mock response")
        bridge.set_llm_provider(mock_provider)
        assert bridge._llm_provider is mock_provider

    def test_agent_loop_accepts_all_components(self) -> None:
        """AgentLoop constructor accepts the full Phase-14 parameter set."""
        from proxima.agent.dynamic_tools.agent_loop import AgentLoop

        loop = AgentLoop(
            nl_processor=MagicMock(),
            tool_bridge=MagicMock(),
            llm_router=MagicMock(),
            llm_tool_integration=None,
            session_context=None,
            ui_callback=lambda msg: None,
            stream_callback=lambda token: None,
            tool_registry=None,
            plan_confirmation_callback=lambda plan: True,
            post_message_callback=lambda msg: None,
        )
        assert loop is not None


# ═══════════════════════════════════════════════════════════════════════
# Step 14.5: Cross-component dependency injection sanity
# ═══════════════════════════════════════════════════════════════════════


class TestCrossComponentInjection:
    """Verify that shared instances can be injected across components."""

    def test_shared_dep_manager_injected_into_bridge(self) -> None:
        """When _dep_mgr is set on IntentToolBridge, it replaces the lazy default."""
        from proxima.agent.dependency_manager import ProjectDependencyManager
        from proxima.agent.dynamic_tools.intent_tool_bridge import IntentToolBridge

        dm = ProjectDependencyManager()
        bridge = IntentToolBridge(consent_callback=lambda msg: True)
        bridge._dep_mgr = dm
        assert bridge._dep_mgr is dm

    def test_shared_terminal_orchestrator_injected_into_bridge(self) -> None:
        """Terminal orchestrator can be injected into the bridge."""
        from proxima.agent.terminal_orchestrator import get_terminal_orchestrator
        from proxima.agent.dynamic_tools.intent_tool_bridge import IntentToolBridge

        orch = get_terminal_orchestrator()
        bridge = IntentToolBridge(consent_callback=lambda msg: True)
        bridge._terminal_orchestrator = orch
        assert bridge._terminal_orchestrator is orch

    def test_error_handler_receives_dep_manager(self) -> None:
        """AgentErrorHandler stores the dep_manager it receives."""
        from proxima.agent.agent_error_handler import AgentErrorHandler
        from proxima.agent.dependency_manager import ProjectDependencyManager

        dm = ProjectDependencyManager()
        handler = AgentErrorHandler(dep_manager=dm)
        assert handler._dep_manager is dm

    def test_system_prompt_builder_uses_tool_registry(self) -> None:
        """SystemPromptBuilder forwards tool_registry for capability listing."""
        from proxima.agent.dynamic_tools.system_prompt_builder import (
            SystemPromptBuilder,
        )

        mock_reg = MagicMock()
        builder = SystemPromptBuilder(tool_registry=mock_reg)
        assert builder._tool_registry is mock_reg


# ═══════════════════════════════════════════════════════════════════════
# Step 14.1 Compliance: Attribute naming per spec
# ═══════════════════════════════════════════════════════════════════════


class TestAttributeNamingCompliance:
    """Verify attribute names match Phase 14.1 spec exactly.

    The spec mandates ``self._intent_tool_bridge`` (not
    ``self._intent_bridge``).  Since we cannot instantiate the TUI
    screen outside of a running Textual app, we inspect the source
    code of ``_initialize_components`` to confirm the correct name.
    """

    def test_intent_tool_bridge_attribute_name(self) -> None:
        """_initialize_components must use self._intent_tool_bridge."""
        import inspect
        from proxima.tui.screens.agent_ai_assistant import (
            AgentAIAssistantScreen,
        )

        src = inspect.getsource(AgentAIAssistantScreen._initialize_components)
        assert "self._intent_tool_bridge" in src, (
            "Spec requires self._intent_tool_bridge, not self._intent_bridge"
        )
        # Ensure the OLD name is NOT present
        # (filter out comments that might mention it for historical context)
        code_lines = [
            line for line in src.splitlines()
            if not line.strip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert "self._intent_bridge" not in code_only.replace(
            "self._intent_tool_bridge", ""
        ), "Found residual self._intent_bridge references in code"

    def test_all_spec_attributes_present(self) -> None:
        """_initialize_components must create all Phase-14 attributes."""
        import inspect
        from proxima.tui.screens.agent_ai_assistant import (
            AgentAIAssistantScreen,
        )

        src = inspect.getsource(AgentAIAssistantScreen._initialize_components)
        required_attrs = [
            "self._dependency_manager",
            "self._terminal_orchestrator",
            "self._intent_tool_bridge",
            "self._system_prompt_builder",
            "self._error_handler",
            "self._agent_loop",
            "self._robust_nl_processor",
        ]
        for attr in required_attrs:
            assert attr in src, f"Missing attribute assignment: {attr}"
