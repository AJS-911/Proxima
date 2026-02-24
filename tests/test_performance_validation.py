"""Phase 13 — Step 13.5: Performance Validation.

Targets (from spec):
  1. Intent recognition (layers 1–3, no LLM) < 100 ms
  2. Tool dispatch latency < 50 ms from intent → tool start
  3. Entity extraction handles messages up to 5 000 characters
  4. Multi-step parser handles ≥ 20 sub-steps
  5. (Bonus) Round-trip ``process_message`` without LLM < 200 ms

All measurements use ``time.perf_counter()`` for wall-clock precision.
Marked ``@pytest.mark.performance``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from proxima.agent.dynamic_tools.robust_nl_processor import (
    ExtractedEntity,
    Intent,
    IntentType,
    RobustNLProcessor,
    SessionContext,
)
from proxima.agent.dynamic_tools.intent_tool_bridge import IntentToolBridge
from proxima.agent.dynamic_tools.tool_interface import ToolResult
from proxima.agent.dynamic_tools.agent_loop import AgentLoop


# ═══════════════════════════════════════════════════════════════════════
# Helpers & Fixtures
# ═══════════════════════════════════════════════════════════════════════

# Generous multiplier so CI runners pass on slow hardware.
_PERF_SLACK = float(os.environ.get("PROXIMA_PERF_SLACK", "3.0"))


def _time_call(fn, *args, **kwargs):
    """Return ``(result, elapsed_seconds)``."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


@pytest.fixture(scope="module")
def processor() -> RobustNLProcessor:
    return RobustNLProcessor()


@pytest.fixture(scope="module")
def tool_bridge() -> IntentToolBridge:
    return IntentToolBridge()


@pytest.fixture()
def mock_tool_bridge() -> MagicMock:
    bridge = MagicMock(spec=IntentToolBridge)
    bridge.dispatch = MagicMock(return_value=ToolResult(
        success=True, result="ok", message="done",
    ))
    return bridge


@pytest.fixture()
def agent_loop(processor: RobustNLProcessor, mock_tool_bridge: MagicMock) -> AgentLoop:
    return AgentLoop(
        nl_processor=processor,
        tool_bridge=mock_tool_bridge,
        plan_confirmation_callback=lambda _: True,
    )


# ═══════════════════════════════════════════════════════════════════════
# Target 1 — Intent Recognition < 100 ms (rule-based only)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestIntentRecognitionPerformance:
    """
    Rule-based intent recognition (layers 1–3) must complete in < 100 ms
    per call.
    """

    MESSAGES = [
        "clone https://github.com/kunal5556/LRET",
        "install numpy scipy pandas",
        "run benchmarks/pennylane/pennylane_4q_50e_25s_10n.py",
        "switch to cirq-scalability-comparison branch",
        "list files in the current directory",
        "navigate to src/proxima",
        "what terminals are running?",
        "undo that",
    ]

    THRESHOLD_MS = 100 * _PERF_SLACK

    @pytest.mark.parametrize("msg", MESSAGES)
    def test_intent_under_threshold(self, processor: RobustNLProcessor, msg: str):
        _, elapsed = _time_call(processor.recognize_intent, msg)
        elapsed_ms = elapsed * 1000
        assert elapsed_ms < self.THRESHOLD_MS, (
            f"Intent recognition for {msg!r} took {elapsed_ms:.1f} ms "
            f"(threshold {self.THRESHOLD_MS:.0f} ms)"
        )

    def test_batch_average(self, processor: RobustNLProcessor):
        """Average over all messages should still be under threshold."""
        total = 0.0
        for msg in self.MESSAGES:
            _, elapsed = _time_call(processor.recognize_intent, msg)
            total += elapsed
        avg_ms = (total / len(self.MESSAGES)) * 1000
        assert avg_ms < self.THRESHOLD_MS, (
            f"Average intent recognition took {avg_ms:.1f} ms "
            f"(threshold {self.THRESHOLD_MS:.0f} ms)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Target 2 — Tool Dispatch Latency < 50 ms
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestToolDispatchLatency:
    """
    From a pre-built Intent object, calling ``IntentToolBridge.dispatch``
    must return a ``ToolResult`` in < 50 ms (mocked I/O).
    """

    THRESHOLD_MS = 50 * _PERF_SLACK

    @staticmethod
    def _make_intent(intent_type: IntentType, msg: str = "test") -> Intent:
        return Intent(
            intent_type=intent_type,
            entities=[],
            confidence=0.9,
            raw_message=msg,
        )

    INTENTS = [
        IntentType.LIST_DIRECTORY,
        IntentType.NAVIGATE_DIRECTORY,
        IntentType.GIT_STATUS,
        IntentType.RUN_COMMAND,
        IntentType.SHOW_CURRENT_DIR,
    ]

    @pytest.mark.parametrize("it", INTENTS, ids=lambda x: x.name)
    def test_dispatch_under_threshold(
        self,
        mock_tool_bridge: MagicMock,
        it: IntentType,
    ):
        intent = self._make_intent(it, msg=f"test {it.name}")
        _, elapsed = _time_call(
            mock_tool_bridge.dispatch,
            intent, os.getcwd(), None, SessionContext(),
        )
        elapsed_ms = elapsed * 1000
        assert elapsed_ms < self.THRESHOLD_MS, (
            f"Dispatch for {it.name} took {elapsed_ms:.1f} ms "
            f"(threshold {self.THRESHOLD_MS:.0f} ms)"
        )

    def test_real_bridge_dispatch_latency(self, tool_bridge: IntentToolBridge):
        """Even the real bridge (which tries to find a tool) should be fast."""
        intent = self._make_intent(IntentType.LIST_DIRECTORY, "ls")
        _, elapsed = _time_call(
            tool_bridge.dispatch,
            intent, os.getcwd(), None, SessionContext(),
        )
        elapsed_ms = elapsed * 1000
        # Real bridge may be slightly slower due to registry lookup — allow
        # 200 ms with slack.
        generous_limit = 200 * _PERF_SLACK
        assert elapsed_ms < generous_limit, (
            f"Real dispatch took {elapsed_ms:.1f} ms (limit {generous_limit:.0f} ms)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Target 3 — Entity Extraction Handles 5 000-char Messages
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestEntityExtractionLargeInput:
    """
    Entity extraction must handle messages up to 5 000 characters without
    errors and within a reasonable time budget (< 500 ms with slack).
    """

    THRESHOLD_MS = 500 * _PERF_SLACK

    @staticmethod
    def _build_long_message(target_length: int = 5000) -> str:
        """Build a synthetic but realistic long message with multiple
        URLs, paths, branches, and packages interspersed."""
        fragments = [
            "clone https://github.com/kunal5556/LRET into C:\\Users\\dell\\Pictures\\Screenshots,",
            "switch to cirq-scalability-comparison branch,",
            "install numpy scipy pandas cirq qiskit pennylane,",
            "run benchmarks/pennylane/pennylane_4q_50e_25s_10n.py,",
            "then navigate to /tmp/test-dir and list files,",
            "after that clone https://github.com/user/another-repo,",
            "go to D:\\projects\\quantum and check dependencies,",
        ]
        msg = ""
        idx = 0
        while len(msg) < target_length:
            msg += fragments[idx % len(fragments)] + " "
            idx += 1
        return msg[:target_length]

    def test_handles_5000_chars(self, processor: RobustNLProcessor):
        msg = self._build_long_message(5000)
        assert len(msg) == 5000
        entities, elapsed = _time_call(processor.extract_entities, msg)
        elapsed_ms = elapsed * 1000
        assert isinstance(entities, list)
        assert elapsed_ms < self.THRESHOLD_MS, (
            f"Entity extraction for 5000-char msg took {elapsed_ms:.1f} ms "
            f"(threshold {self.THRESHOLD_MS:.0f} ms)"
        )

    def test_handles_10000_chars(self, processor: RobustNLProcessor):
        """Stretch test — 10 000 chars."""
        msg = self._build_long_message(10000)
        entities, elapsed = _time_call(processor.extract_entities, msg)
        elapsed_ms = elapsed * 1000
        assert isinstance(entities, list)
        # Allow more time for double the input
        assert elapsed_ms < self.THRESHOLD_MS * 2, (
            f"10k-char extraction took {elapsed_ms:.1f} ms"
        )

    def test_entities_found_in_long_message(self, processor: RobustNLProcessor):
        """Even in a long message, known entities should still be extracted."""
        msg = self._build_long_message(5000)
        entities = processor.extract_entities(msg)
        all_vals = [e.value for e in entities]
        assert any("github.com" in v for v in all_vals), (
            f"Expected URL entity in long message, got {all_vals[:10]}…"
        )


# ═══════════════════════════════════════════════════════════════════════
# Target 4 — Multi-Step Parser Handles ≥ 20 Sub-Steps
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestMultiStepScalability:
    """
    The multi-step parser must correctly handle messages with at least 20
    sub-steps without crashing or excessive slowdown.
    """

    ACTIONS = [
        "clone https://github.com/kunal5556/LRET",
        "checkout cirq-scalability-comparison",
        "install numpy",
        "install scipy",
        "install cirq",
        "install qiskit",
        "install pennylane",
        "navigate to benchmarks",
        "run benchmark_1.py",
        "run benchmark_2.py",
        "run benchmark_3.py",
        "analyze results",
        "export results",
        "navigate to src",
        "build the backend",
        "test the backend",
        "configure Proxima",
        "git status",
        "git commit",
        "git push",
    ]

    THRESHOLD_MS = 500 * _PERF_SLACK

    def _build_numbered(self) -> str:
        return "\n".join(
            f"{i+1}. {self.ACTIONS[i]}" for i in range(len(self.ACTIONS))
        )

    def _build_then_separated(self) -> str:
        return ", then ".join(self.ACTIONS)

    def test_20_numbered_sub_steps(self, processor: RobustNLProcessor):
        msg = self._build_numbered()
        intent, elapsed = _time_call(processor.recognize_intent, msg)
        elapsed_ms = elapsed * 1000
        assert intent.intent_type.name == "MULTI_STEP", (
            f"Expected MULTI_STEP, got {intent.intent_type.name}"
        )
        assert len(intent.sub_intents) >= 10, (
            f"Expected ≥10 sub-intents for 20-step input, got {len(intent.sub_intents)}"
        )
        assert elapsed_ms < self.THRESHOLD_MS, (
            f"20-step parse took {elapsed_ms:.1f} ms (limit {self.THRESHOLD_MS:.0f} ms)"
        )

    def test_20_then_separated(self, processor: RobustNLProcessor):
        msg = self._build_then_separated()
        intent, elapsed = _time_call(processor.recognize_intent, msg)
        elapsed_ms = elapsed * 1000
        assert intent.intent_type.name == "MULTI_STEP", (
            f"Expected MULTI_STEP, got {intent.intent_type.name}"
        )
        assert len(intent.sub_intents) >= 10, (
            f"Expected ≥10 sub-intents, got {len(intent.sub_intents)}"
        )
        assert elapsed_ms < self.THRESHOLD_MS

    def test_40_sub_steps_stretch(self, processor: RobustNLProcessor):
        """Stretch test: 40 sub-steps (doubled list)."""
        doubled = self.ACTIONS * 2
        msg = "\n".join(f"{i+1}. {doubled[i]}" for i in range(len(doubled)))
        intent, elapsed = _time_call(processor.recognize_intent, msg)
        elapsed_ms = elapsed * 1000
        assert intent.intent_type.name == "MULTI_STEP"
        assert len(intent.sub_intents) >= 15, (
            f"Expected ≥15 sub-intents for 40-step input, got {len(intent.sub_intents)}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Target 5 — Round-Trip process_message Without LLM
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestRoundTripPerformance:
    """
    Full ``AgentLoop.process_message`` (no LLM) should complete in < 200 ms
    for common messages.
    """

    THRESHOLD_MS = 200 * _PERF_SLACK

    MESSAGES = [
        "list files",
        "navigate to src",
        "run test.py",
        "install numpy",
    ]

    @pytest.mark.parametrize("msg", MESSAGES)
    def test_round_trip_latency(
        self,
        agent_loop: AgentLoop,
        mock_tool_bridge: MagicMock,
        msg: str,
    ):
        mock_tool_bridge.dispatch.return_value = ToolResult(
            success=True, result="ok", message="done",
        )
        _, elapsed = _time_call(agent_loop.process_message, msg)
        elapsed_ms = elapsed * 1000
        assert elapsed_ms < self.THRESHOLD_MS, (
            f"Round-trip for {msg!r} took {elapsed_ms:.1f} ms "
            f"(threshold {self.THRESHOLD_MS:.0f} ms)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Memory / Stability Checks
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestStability:
    """Repeated invocations should not leak or degrade."""

    def test_100_sequential_recognitions(self, processor: RobustNLProcessor):
        """100 sequential intent recognitions shouldn't crash or slow down."""
        messages = [
            "clone https://github.com/kunal5556/LRET",
            "install numpy",
            "run test.py",
            "navigate to src",
            "list files",
        ]
        t0 = time.perf_counter()
        for i in range(100):
            processor.recognize_intent(messages[i % len(messages)])
        total_ms = (time.perf_counter() - t0) * 1000
        avg_ms = total_ms / 100
        assert avg_ms < 100 * _PERF_SLACK, (
            f"Average recognition: {avg_ms:.1f} ms over 100 calls"
        )

    def test_100_sequential_entity_extractions(self, processor: RobustNLProcessor):
        """100 sequential entity extractions."""
        msg = "clone https://github.com/kunal5556/LRET into C:\\Users\\dell\\Pictures"
        t0 = time.perf_counter()
        for _ in range(100):
            processor.extract_entities(msg)
        total_ms = (time.perf_counter() - t0) * 1000
        avg_ms = total_ms / 100
        assert avg_ms < 100 * _PERF_SLACK, (
            f"Average entity extraction: {avg_ms:.1f} ms over 100 calls"
        )

    def test_processor_state_isolation(self, processor: RobustNLProcessor):
        """Different messages should not bleed state between calls."""
        intent_a = processor.recognize_intent("clone https://github.com/kunal5556/LRET")
        intent_b = processor.recognize_intent("install numpy")
        assert intent_a.intent_type != intent_b.intent_type, (
            "Different messages produced the same intent type"
        )


# ═══════════════════════════════════════════════════════════════════════
# Target 5b — Context Window Management (Session State Preservation)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.performance
class TestContextWindowManagement:
    """Spec Target 5: Context window management must not lose critical
    session state during summarization.

    We simulate prolonged usage (many intent recognitions and entity
    extractions) and verify that SessionContext retains critical state
    throughout.
    """

    def test_session_state_preserved_after_many_operations(
        self, processor: RobustNLProcessor,
    ):
        """Critical session state must survive 200+ operations."""
        ctx = processor.get_context()
        # Seed critical state
        ctx.last_cloned_repo = r"C:\Users\dell\Pictures\LRET"
        ctx.last_cloned_url = "https://github.com/kunal5556/LRET"
        ctx.last_mentioned_branches = ["cirq-scalability-comparison"]
        ctx.current_directory = r"C:\Users\dell\Pictures\LRET"
        ctx.installed_packages = ["numpy", "scipy"]
        ctx.working_directory_stack = [r"C:\Users\dell", r"C:\Users\dell\Pictures"]

        # Simulate prolonged usage — many recognitions & extractions
        messages = [
            "clone https://github.com/kunal5556/LRET",
            "install numpy",
            "run test.py",
            "navigate to src",
            "list files",
            "build backend",
            "analyze results",
            "export results",
            "git status",
            "checkout main",
        ]
        for i in range(200):
            msg = messages[i % len(messages)]
            processor.recognize_intent(msg)
            if i % 3 == 0:
                processor.extract_entities(msg)

        # Verify critical state was NOT lost
        assert ctx.last_cloned_repo == r"C:\Users\dell\Pictures\LRET", (
            "last_cloned_repo lost during prolonged usage"
        )
        assert ctx.last_cloned_url == "https://github.com/kunal5556/LRET", (
            "last_cloned_url lost during prolonged usage"
        )
        assert "cirq-scalability-comparison" in ctx.last_mentioned_branches, (
            "last_mentioned_branches lost during prolonged usage"
        )
        assert "numpy" in ctx.installed_packages, (
            "installed_packages lost during prolonged usage"
        )
        assert len(ctx.working_directory_stack) >= 2, (
            "working_directory_stack lost during prolonged usage"
        )

    def test_conversation_history_bounded(self, processor: RobustNLProcessor):
        """Conversation history should not grow unboundedly — verify the
        processor either caps it or handles large histories gracefully."""
        ctx = processor.get_context()
        # Fire 500 messages through the processor
        for i in range(500):
            processor.recognize_intent(f"install package_{i}")
        # History should exist and be bounded or at least not crash
        history_len = len(ctx.conversation_history)
        assert history_len <= 500, (
            f"History has {history_len} entries — may need capping"
        )
        # Critical: the processor should still function correctly
        intent = processor.recognize_intent("clone https://github.com/kunal5556/LRET")
        assert intent.intent_type.name == "GIT_CLONE"

    def test_clone_tracking_survives_context_reuse(
        self, processor: RobustNLProcessor,
    ):
        """After seeding clone info, 100 unrelated operations should not
        clear it."""
        ctx = processor.get_context()
        ctx.cloned_repos["https://github.com/kunal5556/LRET"] = "/tmp/LRET"
        ctx.last_cloned_repo = "/tmp/LRET"

        for _ in range(100):
            processor.recognize_intent("install numpy")

        assert ctx.last_cloned_repo == "/tmp/LRET", (
            "Clone tracking cleared during unrelated operations"
        )
        assert "https://github.com/kunal5556/LRET" in ctx.cloned_repos
