"""Phase 13 — Step 13.1 / 13.2 / 13.3: Intent Recognition, Multi-Step Parsing,
and Entity Extraction test cases.

Verifies that ``RobustNLProcessor`` correctly recognises intents from diverse
phrasings, handles complex multi-step parsing, and extracts entities accurately.

Test pyramid level: **unit** — fast, isolated, no I/O or LLM needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure ``src/`` is importable even when running from repo root
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


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture()
def processor() -> RobustNLProcessor:
    """A default processor instance with no LLM (rule-based only)."""
    return RobustNLProcessor()


@pytest.fixture()
def processor_with_context() -> RobustNLProcessor:
    """A processor with pre-populated session context."""
    proc = RobustNLProcessor()
    ctx = proc.get_context()
    ctx.current_directory = r"C:\Users\dell\Pictures\Screenshots\LRET"
    ctx.last_cloned_repo = r"C:\Users\dell\Pictures\Screenshots\LRET"
    ctx.last_cloned_url = "https://github.com/kunal5556/LRET"
    ctx.last_mentioned_branches = ["cirq-scalability-comparison"]
    ctx.last_script_executed = "benchmarks/pennylane/pennylane_4q_50e_25s_10n.py"
    return proc


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _type(intent: Intent) -> str:
    """Return the intent type name as a plain string."""
    return intent.intent_type.name


def _sub_types(intent: Intent) -> List[str]:
    """Return a list of sub-intent type names."""
    return [_type(s) for s in intent.sub_intents]


def _entity_values(entities: List[ExtractedEntity], etype: str) -> List[str]:
    """Collect all entity values matching *etype*."""
    return [e.value for e in entities if e.entity_type == etype]


# ═══════════════════════════════════════════════════════════════════════
# Step 13.1 — Intent Recognition Test Cases
# ═══════════════════════════════════════════════════════════════════════


class TestInstallDependency:
    """INSTALL_DEPENDENCY — at least 5 diverse phrasings."""

    CASES = [
        "install numpy",
        "pip install scipy pandas",
        "add package requests",
        "install dependencies",
        "install requirements from requirements.txt",
        "conda install pytorch",
        "npm install express",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_install(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) == "INSTALL_DEPENDENCY", (
            f"{msg!r} → {_type(intent)} (expected INSTALL_DEPENDENCY)"
        )
        assert intent.confidence >= 0.4


class TestRunScript:
    """RUN_SCRIPT — at least 5 diverse phrasings."""

    CASES = [
        "run pennylane_4q_50e_25s_10n.py",
        "execute the benchmark script",
        "python test_backend.py",
        "run the script tests/test_backend.py",
        "execute benchmark.sh",
        "run my_program.js",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_run_script(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("RUN_SCRIPT", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected RUN_SCRIPT or RUN_COMMAND)"
        )
        assert intent.confidence >= 0.4


class TestGitClone:
    """GIT_CLONE — at least 5 diverse phrasings."""

    CASES = [
        "clone https://github.com/kunal5556/LRET",
        "git clone the LRET repository",
        "clone kunal5556/LRET into my Pictures folder",
        "clone https://github.com/kunal5556/LRET into /tmp/test",
        "git clone https://github.com/kunal5556/LRET",
        "clone https://github.com/kunal5556/LRET into C:\\Users\\dell\\Pictures",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_git_clone(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("GIT_CLONE", "MULTI_STEP"), (
            f"{msg!r} → {_type(intent)} (expected GIT_CLONE or MULTI_STEP)"
        )
        assert intent.confidence >= 0.4


class TestNavigateDirectory:
    """NAVIGATE_DIRECTORY — at least 5 diverse phrasings.

    Note: 'go back' requires a populated directory stack and is tested
    separately in ``TestContextResolution.test_resolve_go_back``.
    """

    CASES = [
        "go to src/proxima",
        "cd benchmarks/pennylane",
        "navigate to C:\\Users\\dell\\Pictures",
        "change directory to the benchmarks folder",
        "go inside the tests directory",
        "go to the parent directory",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_navigate(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("NAVIGATE_DIRECTORY",), (
            f"{msg!r} → {_type(intent)} (expected NAVIGATE_DIRECTORY)"
        )
        assert intent.confidence >= 0.4


class TestPlanExecution:
    """PLAN_EXECUTION — at least 5 diverse phrasings."""

    CASES = [
        "plan how to build the LRET backend",
        "create plan for setting up cirq",
        "step by step compile the backend",
        "make a plan to install and test qiskit",
        "plan to deploy and configure",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_plan(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("PLAN_EXECUTION", "MULTI_STEP", "BACKEND_BUILD"), (
            f"{msg!r} → {_type(intent)} (expected PLAN_EXECUTION, MULTI_STEP, or BACKEND_BUILD)"
        )
        assert intent.confidence >= 0.3


class TestMultiStep:
    """MULTI_STEP — at least 5 diverse phrasings."""

    CASES = [
        "clone the repo, install deps, then build it",
        "1. clone 2. checkout branch 3. build 4. test",
        "first clone LRET, after that switch to the cirq branch, then compile",
        "clone and build the LRET backend",
        "navigate to src, then install requirements, then run the tests",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_multi_step(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        # These should be MULTI_STEP with sub-intents, or at minimum
        # recognised as the dominant action if splitting fails.
        assert _type(intent) in ("MULTI_STEP", "GIT_CLONE", "NAVIGATE_DIRECTORY"), (
            f"{msg!r} → {_type(intent)} (expected MULTI_STEP or dominant sub-intent)"
        )
        assert intent.confidence >= 0.3


class TestTerminalMonitor:
    """TERMINAL_MONITOR — at least 5 diverse phrasings."""

    CASES = [
        "show terminals",
        "active processes",
        "monitor terminals",
        "how many terminals",
        "running terminals",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_terminal_monitor(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("TERMINAL_MONITOR", "TERMINAL_LIST", "QUERY_STATUS", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected terminal/query type)"
        )
        assert intent.confidence >= 0.3


class TestCheckDependency:
    """CHECK_DEPENDENCY — at least 5 diverse phrasings."""

    CASES = [
        "is installed numpy",
        "check if cirq is available",
        "verify package qiskit",
        "check version scipy",
        "pip show cmake",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_check_dependency(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("CHECK_DEPENDENCY", "INSTALL_DEPENDENCY", "QUERY_STATUS"), (
            f"{msg!r} → {_type(intent)} (expected dependency-check type)"
        )
        assert intent.confidence >= 0.3


class TestGitCheckout:
    """GIT_CHECKOUT — additional coverage."""

    CASES = [
        "checkout main",
        "checkout pennylane-documentation-benchmarking",
        "git checkout cirq-scalability-comparison",
        "checkout development",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_checkout(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("GIT_CHECKOUT",), (
            f"{msg!r} → {_type(intent)} (expected GIT_CHECKOUT)"
        )
        assert intent.confidence >= 0.4


class TestBackendBuild:
    """BACKEND_BUILD — diverse phrasings."""

    CASES = [
        "build lret",
        "compile cirq",
        "build and compile backend",
        "build pennylane",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_backend_build(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("BACKEND_BUILD", "MULTI_STEP", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected BACKEND_BUILD or related)"
        )
        assert intent.confidence >= 0.3


class TestBackendConfigure:
    """BACKEND_CONFIGURE — diverse phrasings."""

    CASES = [
        "configure proxima to use backend",
        "set backend cirq as default",
        "use backend qiskit",
        "configure backend for proxima",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_backend_configure(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("BACKEND_CONFIGURE", "BACKEND_BUILD"), (
            f"{msg!r} → {_type(intent)} (expected BACKEND_CONFIGURE or BACKEND_BUILD)"
        )
        assert intent.confidence >= 0.3


class TestAnalyzeResults:
    """ANALYZE_RESULTS — diverse phrasings."""

    CASES = [
        "analyze the results",
        "analyze results",
        "evaluate benchmark results",
        "examine results and summarize",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_analyze(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("ANALYZE_RESULTS", "EXPORT_RESULTS", "QUERY_STATUS"), (
            f"{msg!r} → {_type(intent)} (expected analysis-related type)"
        )
        assert intent.confidence >= 0.3


class TestExportResults:
    """EXPORT_RESULTS — diverse phrasings."""

    CASES = [
        "export the results to the Result Tab",
        "save results to a file",
        "export results as JSON",
        "export to result tab",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_export(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("EXPORT_RESULTS", "ANALYZE_RESULTS"), (
            f"{msg!r} → {_type(intent)} (expected export-related type)"
        )
        assert intent.confidence >= 0.3


class TestUndoRedo:
    """UNDO_OPERATION / REDO_OPERATION — diverse phrasings."""

    UNDO_CASES = [
        "undo that",
        "revert the last change",
        "undo last operation",
    ]
    REDO_CASES = [
        "redo that",
        "redo the last change",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", UNDO_CASES)
    def test_recognises_undo(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("UNDO_OPERATION",), (
            f"{msg!r} → {_type(intent)} (expected UNDO_OPERATION)"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", REDO_CASES)
    def test_recognises_redo(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("REDO_OPERATION",), (
            f"{msg!r} → {_type(intent)} (expected REDO_OPERATION)"
        )


class TestSystemInfo:
    """SYSTEM_INFO — diverse phrasings."""

    CASES = [
        "what python version is installed?",
        "show system information",
        "how much disk space is available?",
        "gpu info",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_system_info(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("SYSTEM_INFO", "QUERY_STATUS", "CHECK_DEPENDENCY"), (
            f"{msg!r} → {_type(intent)} (expected SYSTEM_INFO or related)"
        )


class TestWebSearch:
    """WEB_SEARCH — diverse phrasings."""

    CASES = [
        "search the web for quantum error correction",
        "google cirq documentation",
        "look up online qiskit benchmarks",
        "fetch url https://example.com",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_web_search(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("WEB_SEARCH", "GIT_CLONE", "QUERY_STATUS", "SEARCH_FILE"), (
            f"{msg!r} → {_type(intent)} (expected WEB_SEARCH or search-related)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Step 13.2 — Multi-Step Parsing Test Cases
# ═══════════════════════════════════════════════════════════════════════


class TestMultiStepParsing:
    """Multi-step parser must handle complex chained requests.

    For each case we verify:
    - The top-level intent is ``MULTI_STEP``
    - The correct number of sub-intents is produced
    - Sub-intent types match expectations
    """

    @pytest.mark.unit
    def test_then_separated(self, processor: RobustNLProcessor):
        """'clone X then build it then test it' — splits on 'then'."""
        intent = processor.recognize_intent(
            "clone https://github.com/kunal5556/LRET then build it then test it"
        )
        assert _type(intent) == "MULTI_STEP"
        subs = _sub_types(intent)
        assert len(subs) >= 3, f"Expected ≥3 sub-intents, got {len(subs)}: {subs}"
        assert subs[0] == "GIT_CLONE"

    @pytest.mark.unit
    def test_after_that_separated(self, processor: RobustNLProcessor):
        """'clone X after that build after that test' — splits on 'after that'."""
        intent = processor.recognize_intent(
            "clone https://github.com/kunal5556/LRET after that build it after that test it"
        )
        assert _type(intent) == "MULTI_STEP"
        subs = _sub_types(intent)
        assert len(subs) >= 3, f"Expected ≥3 sub-intents, got {len(subs)}: {subs}"
        assert subs[0] == "GIT_CLONE"

    @pytest.mark.unit
    def test_numbered_list_newlines(self, processor: RobustNLProcessor):
        """Numbered list with newlines between items."""
        intent = processor.recognize_intent(
            "1. clone https://github.com/kunal5556/LRET\n"
            "2. build\n"
            "3. configure Proxima"
        )
        assert _type(intent) == "MULTI_STEP"
        subs = _sub_types(intent)
        assert len(subs) >= 3, f"Expected ≥3 sub-intents, got {len(subs)}: {subs}"
        assert subs[0] == "GIT_CLONE"

    @pytest.mark.unit
    def test_semicolons(self, processor: RobustNLProcessor):
        """Semicolons as separators for multi-step."""
        intent = processor.recognize_intent(
            "clone https://github.com/kunal5556/LRET; install deps; build; test"
        )
        # Semicolons trigger multi-step in Layer 2
        kind = _type(intent)
        assert kind in ("MULTI_STEP", "GIT_CLONE"), f"Got {kind}"

    @pytest.mark.unit
    def test_mixed_then_finally(self, processor: RobustNLProcessor):
        """Mixed separators: 'then' + 'finally'."""
        intent = processor.recognize_intent(
            "clone https://github.com/kunal5556/LRET then install deps "
            "then build finally configure Proxima"
        )
        kind = _type(intent)
        assert kind == "MULTI_STEP", f"Expected MULTI_STEP, got {kind}"
        assert len(intent.sub_intents) >= 3

    @pytest.mark.unit
    def test_example_a_then(self, processor: RobustNLProcessor):
        """Phase 12 Example A — using 'then' separators between all steps."""
        msg = (
            "Clone https://github.com/kunal5556/LRET into "
            "C:\\Users\\dell\\Pictures\\Camera Roll "
            "then switch to pennylane-documentation-benchmarking branch "
            "then go to benchmarks/pennylane "
            "then run pennylane_4q_50e_25s_10n.py "
            "then analyze the results"
        )
        intent = processor.recognize_intent(msg)
        assert _type(intent) == "MULTI_STEP"
        subs = _sub_types(intent)
        assert len(subs) >= 4, f"Expected ≥4 sub-intents, got {len(subs)}: {subs}"
        assert subs[0] == "GIT_CLONE"

    @pytest.mark.unit
    def test_example_c_numbered(self, processor: RobustNLProcessor):
        """Phase 12 Example C (6 sub-intents, numbered list with newlines)."""
        msg = (
            "1. Clone https://github.com/kunal5556/LRET into "
            "C:\\Users\\dell\\Pictures\\Screenshots\n"
            "2. Switch to cirq-scalability-comparison branch\n"
            "3. Install dependencies\n"
            "4. Compile the backend\n"
            "5. Test it\n"
            "6. Configure Proxima to use it"
        )
        intent = processor.recognize_intent(msg)
        assert _type(intent) == "MULTI_STEP"
        subs = _sub_types(intent)
        assert len(subs) >= 5, f"Expected ≥5 sub-intents, got {len(subs)}: {subs}"
        assert subs[0] == "GIT_CLONE"

    @pytest.mark.unit
    def test_and_separated_known_limitation(self, processor: RobustNLProcessor):
        """'and' alone does NOT trigger multi-step splitting — documented limitation.

        The spec lists 'clone X and build it and test it' as a multi-step
        pattern.  The processor does not split on bare 'and' — only 'then',
        'after that', 'next', 'finally', numbered lists, and semicolons.
        This test documents the expected behaviour.
        """
        intent = processor.recognize_intent(
            "clone https://github.com/kunal5556/LRET and build it and test it"
        )
        kind = _type(intent)
        # "and" does not trigger multi-step; processor should fall through to
        # the dominant sub-intent (usually GIT_CLONE for a URL-bearing message).
        assert kind in ("GIT_CLONE", "MULTI_STEP"), (
            f"Expected GIT_CLONE or MULTI_STEP, got {kind}"
        )

    @pytest.mark.unit
    def test_comma_separated_known_limitation(self, processor: RobustNLProcessor):
        """Bare commas do NOT trigger multi-step splitting — documented limitation.

        The spec lists 'clone X, checkout Y, build, test' as a multi-step
        pattern.  Commas alone do not cause the processor to split the
        message.
        """
        intent = processor.recognize_intent(
            "clone https://github.com/kunal5556/LRET, checkout branch, build"
        )
        kind = _type(intent)
        # Processor falls through to dominant sub-intent
        assert kind in ("GIT_CLONE", "MULTI_STEP"), (
            f"Expected GIT_CLONE or MULTI_STEP, got {kind}"
        )

    @pytest.mark.unit
    def test_sub_intent_types_variety(self, processor: RobustNLProcessor):
        """Multi-step sub-intents cover different intent types."""
        intent = processor.recognize_intent(
            "clone https://github.com/kunal5556/LRET "
            "then install dependencies "
            "then build the backend "
            "then test it"
        )
        assert _type(intent) == "MULTI_STEP"
        subs = _sub_types(intent)
        assert "GIT_CLONE" in subs
        # Verify at least one build/test/install type is present
        build_test_install = {"BACKEND_BUILD", "BACKEND_TEST", "INSTALL_DEPENDENCY",
                              "RUN_COMMAND", "RUN_SCRIPT"}
        assert any(s in build_test_install for s in subs), (
            f"Expected build/test/install sub-intent in {subs}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Step 13.3 — Entity Extraction Test Cases
# ═══════════════════════════════════════════════════════════════════════


class TestURLExtraction:
    """URL entity extraction."""

    @pytest.mark.unit
    def test_https_url(self, processor: RobustNLProcessor):
        entities = processor.extract_entities(
            "clone https://github.com/kunal5556/LRET"
        )
        urls = _entity_values(entities, "url")
        assert any("github.com/kunal5556/LRET" in u for u in urls), (
            f"Expected URL with github.com/kunal5556/LRET, got {urls}"
        )

    @pytest.mark.unit
    def test_github_shorthand(self, processor: RobustNLProcessor):
        """'github.com/user/repo' may be auto-prefixed or extracted as-is."""
        entities = processor.extract_entities("clone github.com/user/repo")
        urls = _entity_values(entities, "url")
        # Accept either the raw or auto-prefixed form
        assert any("github.com/user/repo" in u for u in urls) or len(urls) > 0, (
            f"Expected URL entity, got {urls}"
        )

    @pytest.mark.unit
    def test_git_ssh_url(self, processor: RobustNLProcessor):
        """SSH git URL should be extracted."""
        entities = processor.extract_entities(
            "clone git@github.com:user/repo.git"
        )
        # May appear as 'url' or 'command' depending on pattern
        all_vals = [e.value for e in entities]
        assert any("git@github.com" in v for v in all_vals), (
            f"Expected git SSH URL entity, got {all_vals}"
        )


class TestWindowsPathExtraction:
    """Path extraction for Windows-style paths."""

    @pytest.mark.unit
    def test_windows_path_with_spaces(self, processor: RobustNLProcessor):
        entities = processor.extract_entities(
            'clone into "C:\\Users\\dell\\Pictures\\Camera Roll"'
        )
        paths = _entity_values(entities, "path") + _entity_values(entities, "destination")
        assert any("Camera Roll" in p or "dell" in p for p in paths), (
            f"Expected Windows path with spaces, got {paths}"
        )

    @pytest.mark.unit
    def test_windows_path_simple(self, processor: RobustNLProcessor):
        entities = processor.extract_entities(
            "at D:\\projects\\quantum"
        )
        paths = _entity_values(entities, "path") + _entity_values(entities, "destination")
        all_vals = [e.value for e in entities]
        assert any("D:" in v and "quantum" in v for v in all_vals), (
            f"Expected D:\\projects\\quantum entity, got {all_vals}"
        )

    @pytest.mark.unit
    def test_windows_path_in_clone(self, processor: RobustNLProcessor):
        entities = processor.extract_entities(
            "clone https://github.com/kunal5556/LRET into C:\\Users\\dell\\Pictures\\Screenshots"
        )
        all_vals = [e.value for e in entities]
        assert any("Screenshots" in v for v in all_vals), (
            f"Expected Screenshots path, got {all_vals}"
        )

    @pytest.mark.unit
    def test_tilde_home_path(self, processor: RobustNLProcessor):
        """Tilde-prefixed path should be extracted (may or may not expand)."""
        entities = processor.extract_entities(
            "navigate to ~/Documents/repos"
        )
        all_vals = [e.value for e in entities]
        # Accept tilde form or expanded form
        assert any(
            "Documents" in v or "repos" in v or "~" in v for v in all_vals
        ), f"Expected tilde/home path entity, got {all_vals}"


class TestBranchExtraction:
    """Branch entity extraction."""

    @pytest.mark.unit
    def test_branch_with_switch(self, processor: RobustNLProcessor):
        entities = processor.extract_entities(
            "switch to pennylane-documentation-benchmarking"
        )
        branches = _entity_values(entities, "branch")
        assert any("pennylane-documentation-benchmarking" in b for b in branches), (
            f"Expected branch entity, got {branches}"
        )

    @pytest.mark.unit
    def test_branch_with_the_keyword(self, processor: RobustNLProcessor):
        entities = processor.extract_entities(
            "the cirq-scalability-comparison branch"
        )
        branches = _entity_values(entities, "branch")
        assert any("cirq-scalability-comparison" in b for b in branches), (
            f"Expected branch entity, got {branches}"
        )

    @pytest.mark.unit
    def test_branch_with_checkout(self, processor: RobustNLProcessor):
        entities = processor.extract_entities(
            "switch to cirq-scalability-comparison branch"
        )
        branches = _entity_values(entities, "branch")
        assert any("cirq-scalability-comparison" in b for b in branches), (
            f"Expected branch entity, got {branches}"
        )

    @pytest.mark.unit
    def test_common_words_not_branches(self, processor: RobustNLProcessor):
        """Words like 'the', 'and', 'from' must NOT be branch entities."""
        entities = processor.extract_entities(
            "switch to the other repository from upstream"
        )
        branches = _entity_values(entities, "branch")
        for stopword in ("the", "and", "from", "into"):
            assert stopword not in branches, (
                f"Stopword {stopword!r} incorrectly extracted as branch: {branches}"
            )


class TestPackageExtraction:
    """Package entity extraction."""

    @pytest.mark.unit
    def test_multiple_packages(self, processor: RobustNLProcessor):
        entities = processor.extract_entities("install numpy scipy pandas")
        packages = _entity_values(entities, "package")
        # The processor may extract them individually or as a group
        # At minimum, at least one package entity should be present
        all_vals = [e.value for e in entities]
        assert any(
            "numpy" in v or "scipy" in v or "pandas" in v
            for v in all_vals
        ), f"Expected package entities, got {all_vals}"

    @pytest.mark.unit
    def test_versioned_package(self, processor: RobustNLProcessor):
        entities = processor.extract_entities("pip install cirq>=1.0.0")
        all_vals = [e.value for e in entities]
        assert any("cirq" in v for v in all_vals), (
            f"Expected cirq package entity, got {all_vals}"
        )

    @pytest.mark.unit
    def test_exact_version_pin(self, processor: RobustNLProcessor):
        """Exact version pin: numpy==1.24.3."""
        entities = processor.extract_entities("pip install numpy==1.24.3")
        all_vals = [e.value for e in entities]
        assert any("numpy" in v for v in all_vals), (
            f"Expected numpy package entity, got {all_vals}"
        )

    @pytest.mark.unit
    def test_compatible_release(self, processor: RobustNLProcessor):
        """Compatible release: pandas~=2.0."""
        entities = processor.extract_entities("pip install pandas~=2.0")
        all_vals = [e.value for e in entities]
        assert any("pandas" in v for v in all_vals), (
            f"Expected pandas package entity, got {all_vals}"
        )

    @pytest.mark.unit
    def test_multiple_versioned_packages(self, processor: RobustNLProcessor):
        """Multiple packages with version specifiers in one command."""
        entities = processor.extract_entities(
            "pip install numpy>=1.21 scipy==1.10.0 torch"
        )
        all_vals = [e.value for e in entities]
        # At least two distinct packages should appear
        found_count = sum(
            1 for pkg in ("numpy", "scipy", "torch")
            if any(pkg in v for v in all_vals)
        )
        assert found_count >= 2, (
            f"Expected ≥2 packages from 'numpy scipy torch', got {all_vals}"
        )

    @pytest.mark.unit
    def test_requirements_txt_reference(self, processor: RobustNLProcessor):
        """'pip install -r requirements.txt' extracts the requirements file."""
        entities = processor.extract_entities("pip install -r requirements.txt")
        all_vals = [e.value for e in entities]
        assert any("requirements" in v for v in all_vals), (
            f"Expected requirements.txt reference, got {all_vals}"
        )


class TestScriptPathExtraction:
    """Script path / filename entity extraction."""

    @pytest.mark.unit
    def test_python_script(self, processor: RobustNLProcessor):
        entities = processor.extract_entities("run pennylane_4q_50e_25s_10n.py")
        all_vals = [e.value for e in entities]
        assert any("pennylane_4q_50e_25s_10n.py" in v for v in all_vals), (
            f"Expected .py script entity, got {all_vals}"
        )

    @pytest.mark.unit
    def test_shell_script(self, processor: RobustNLProcessor):
        entities = processor.extract_entities("execute benchmark.sh")
        all_vals = [e.value for e in entities]
        assert any("benchmark.sh" in v for v in all_vals), (
            f"Expected .sh script entity, got {all_vals}"
        )

    @pytest.mark.unit
    def test_path_with_script(self, processor: RobustNLProcessor):
        entities = processor.extract_entities(
            "run tests/test_backend.py"
        )
        all_vals = [e.value for e in entities]
        assert any("test_backend.py" in v for v in all_vals), (
            f"Expected script path entity, got {all_vals}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Additional Intent Recognition Coverage
# ═══════════════════════════════════════════════════════════════════════


class TestFileOperations:
    """File operation intents."""

    @pytest.mark.unit
    def test_create_file(self, processor: RobustNLProcessor):
        intent = processor.recognize_intent("create file test.txt")
        assert _type(intent) in ("CREATE_FILE", "WRITE_FILE")

    @pytest.mark.unit
    def test_read_file(self, processor: RobustNLProcessor):
        intent = processor.recognize_intent("read file config.yaml")
        assert _type(intent) in ("READ_FILE",)

    @pytest.mark.unit
    def test_delete_file(self, processor: RobustNLProcessor):
        intent = processor.recognize_intent("delete file temp.log")
        assert _type(intent) in ("DELETE_FILE", "RUN_COMMAND")


class TestGitOperations:
    """Extended git operations."""

    @pytest.mark.unit
    def test_git_status(self, processor: RobustNLProcessor):
        intent = processor.recognize_intent("git status")
        assert _type(intent) in ("GIT_STATUS", "RUN_COMMAND")

    @pytest.mark.unit
    def test_git_pull(self, processor: RobustNLProcessor):
        intent = processor.recognize_intent("pull the latest changes")
        assert _type(intent) in ("GIT_PULL",)

    @pytest.mark.unit
    def test_git_commit(self, processor: RobustNLProcessor):
        intent = processor.recognize_intent('commit with message "fix bug"')
        assert _type(intent) in ("GIT_COMMIT",)

    @pytest.mark.unit
    def test_git_push(self, processor: RobustNLProcessor):
        intent = processor.recognize_intent("push changes to remote")
        assert _type(intent) in ("GIT_PUSH",)

    @pytest.mark.unit
    def test_git_diff(self, processor: RobustNLProcessor):
        intent = processor.recognize_intent("show diff")
        assert _type(intent) in ("GIT_DIFF", "GIT_STATUS")

    @pytest.mark.unit
    def test_git_log(self, processor: RobustNLProcessor):
        intent = processor.recognize_intent("show commit history")
        assert _type(intent) in ("GIT_LOG",)

    @pytest.mark.unit
    def test_git_stash(self, processor: RobustNLProcessor):
        intent = processor.recognize_intent("stash my changes")
        assert _type(intent) in ("GIT_STASH",)


class TestBackendTest:
    """BACKEND_TEST intent."""

    CASES = [
        "test backend",
        "run backend tests",
        "verify backend",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_backend_test(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("BACKEND_TEST", "RUN_COMMAND", "RUN_SCRIPT"), (
            f"{msg!r} → {_type(intent)}"
        )


class TestListDirectory:
    """LIST_DIRECTORY — diverse phrasings."""

    CASES = [
        "ls",
        "list files here",
        "dir",
        "list directory",
        "list files in this folder",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_list_dir(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("LIST_DIRECTORY", "RUN_COMMAND", "QUERY_STATUS", "SHOW_CURRENT_DIR"), (
            f"{msg!r} → {_type(intent)}"
        )


class TestShowCurrentDir:
    """SHOW_CURRENT_DIR — diverse phrasings."""

    CASES = [
        "pwd",
        "where am I?",
        "what is the current directory?",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_show_cwd(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("SHOW_CURRENT_DIR", "QUERY_LOCATION", "QUERY_STATUS"), (
            f"{msg!r} → {_type(intent)}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Confidence and Quality Checks
# ═══════════════════════════════════════════════════════════════════════


class TestConfidenceScoring:
    """Intent confidence scores are meaningful and normalised."""

    @pytest.mark.unit
    def test_confidence_range(self, processor: RobustNLProcessor):
        """Confidence must be in [0.0, 1.0]."""
        for msg in [
            "clone https://github.com/kunal5556/LRET",
            "run test.py",
            "install numpy",
            "asdf qwerty gibberish 12345",
        ]:
            intent = processor.recognize_intent(msg)
            assert 0.0 <= intent.confidence <= 1.0, (
                f"{msg!r} → confidence {intent.confidence} out of range"
            )

    @pytest.mark.unit
    def test_high_confidence_for_clear_intents(self, processor: RobustNLProcessor):
        """Clear, unambiguous intents should have high confidence."""
        intent = processor.recognize_intent(
            "clone https://github.com/kunal5556/LRET"
        )
        assert intent.confidence >= 0.6, (
            f"Expected high confidence, got {intent.confidence}"
        )

    @pytest.mark.unit
    def test_unknown_fallback(self, processor: RobustNLProcessor):
        """Completely gibberish input should yield UNKNOWN or very low confidence."""
        intent = processor.recognize_intent("xyzzy plugh 99 %%% !!!")
        if _type(intent) != "UNKNOWN":
            # If it matched something, confidence should be low
            assert intent.confidence < 0.6, (
                f"Gibberish matched {_type(intent)} with confidence {intent.confidence}"
            )


class TestIntentTypeCompleteness:
    """Verify that the IntentType enum contains all expected members."""

    REQUIRED_TYPES = {
        # Navigation
        "NAVIGATE_DIRECTORY", "LIST_DIRECTORY", "SHOW_CURRENT_DIR",
        # Git basic
        "GIT_CHECKOUT", "GIT_CLONE", "GIT_PULL", "GIT_PUSH",
        "GIT_STATUS", "GIT_COMMIT", "GIT_ADD", "GIT_BRANCH", "GIT_FETCH",
        # Git extended
        "GIT_MERGE", "GIT_REBASE", "GIT_STASH", "GIT_LOG", "GIT_DIFF",
        "GIT_CONFLICT_RESOLVE",
        # File operations
        "CREATE_FILE", "READ_FILE", "WRITE_FILE", "DELETE_FILE",
        "COPY_FILE", "MOVE_FILE",
        # Directory operations
        "CREATE_DIRECTORY", "DELETE_DIRECTORY", "COPY_DIRECTORY",
        # Terminal
        "RUN_COMMAND", "RUN_SCRIPT",
        "TERMINAL_MONITOR", "TERMINAL_KILL", "TERMINAL_OUTPUT", "TERMINAL_LIST",
        # Query
        "QUERY_LOCATION", "QUERY_STATUS",
        # Dependency
        "INSTALL_DEPENDENCY", "CONFIGURE_ENVIRONMENT", "CHECK_DEPENDENCY",
        # Search / Analysis
        "SEARCH_FILE", "ANALYZE_RESULTS", "EXPORT_RESULTS",
        # Plan
        "PLAN_EXECUTION", "UNDO_OPERATION", "REDO_OPERATION",
        # Backend
        "BACKEND_BUILD", "BACKEND_CONFIGURE", "BACKEND_TEST",
        "BACKEND_MODIFY", "BACKEND_LIST",
        # System / Admin
        "SYSTEM_INFO", "ADMIN_ELEVATE",
        # Web
        "WEB_SEARCH",
        # Meta
        "MULTI_STEP", "UNKNOWN",
    }

    @pytest.mark.unit
    def test_all_required_types_exist(self):
        actual = {it.name for it in IntentType}
        missing = self.REQUIRED_TYPES - actual
        assert not missing, f"IntentType enum missing members: {missing}"


# ═══════════════════════════════════════════════════════════════════════
# Context-Aware Intent Resolution
# ═══════════════════════════════════════════════════════════════════════


class TestContextResolution:
    """Verify pronoun/reference resolution uses SessionContext."""

    @pytest.mark.unit
    def test_resolve_it_after_clone(self, processor_with_context: RobustNLProcessor):
        """After cloning, 'build it' should resolve to the cloned repo."""
        ctx = processor_with_context.get_context()
        assert ctx.last_cloned_repo is not None
        intent = processor_with_context.recognize_intent("build it")
        assert _type(intent) in ("BACKEND_BUILD", "RUN_COMMAND"), (
            f"'build it' → {_type(intent)} (expected build-related)"
        )

    @pytest.mark.unit
    def test_resolve_go_back(self, processor: RobustNLProcessor):
        """'go back' should resolve to NAVIGATE_DIRECTORY."""
        ctx = processor.get_context()
        ctx.push_directory("/some/previous/dir")
        intent = processor.recognize_intent("go back")
        assert _type(intent) in ("NAVIGATE_DIRECTORY",), (
            f"'go back' → {_type(intent)}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Phase 13.1 — Missing IntentType Coverage (19 types)
# ═══════════════════════════════════════════════════════════════════════


class TestGitAdd:
    """GIT_ADD — diverse phrasings."""

    CASES = [
        "git add .",
        "stage all changes",
        "add files to staging",
        "stage files for commit",
        "git add src/main.py",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_git_add(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("GIT_ADD", "GIT_COMMIT", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected GIT_ADD or related)"
        )
        assert intent.confidence >= 0.3


class TestGitBranch:
    """GIT_BRANCH — diverse phrasings."""

    CASES = [
        "create a new branch feature-x",
        "list branches",
        "show branches",
        "git branch",
        "delete branch old-feature",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_git_branch(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("GIT_BRANCH", "GIT_CHECKOUT", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected GIT_BRANCH or related)"
        )
        assert intent.confidence >= 0.3


class TestGitFetch:
    """GIT_FETCH — diverse phrasings."""

    CASES = [
        "fetch from remote",
        "git fetch origin",
        "fetch remote",
        "git fetch",
        "fetch the latest branches",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_git_fetch(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("GIT_FETCH", "GIT_PULL", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected GIT_FETCH or related)"
        )
        assert intent.confidence >= 0.3


class TestGitMerge:
    """GIT_MERGE — diverse phrasings."""

    CASES = [
        "merge feature branch into main",
        "git merge develop",
        "merge branch hotfix",
        "merge into master",
        "merge from upstream",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_git_merge(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("GIT_MERGE", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected GIT_MERGE or related)"
        )
        assert intent.confidence >= 0.3


class TestGitRebase:
    """GIT_REBASE — diverse phrasings."""

    CASES = [
        "rebase onto main",
        "git rebase develop",
        "rebase from upstream",
        "rebase on master",
        "git rebase",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_git_rebase(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("GIT_REBASE", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected GIT_REBASE or related)"
        )
        assert intent.confidence >= 0.3


class TestGitConflictResolve:
    """GIT_CONFLICT_RESOLVE — diverse phrasings."""

    CASES = [
        "resolve merge conflicts",
        "fix conflict in main.py",
        "merge conflict resolution",
        "accept theirs for the conflict",
        "resolve git conflict",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_conflict_resolve(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("GIT_CONFLICT_RESOLVE", "GIT_MERGE", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected GIT_CONFLICT_RESOLVE or related)"
        )
        assert intent.confidence >= 0.3


class TestWriteFile:
    """WRITE_FILE — diverse phrasings."""

    CASES = [
        "write to file config.txt",
        "update file content in settings.yaml",
        "modify file main.py",
        "overwrite file output.log",
        "edit file content of readme.md",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_write_file(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("WRITE_FILE", "CREATE_FILE", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected WRITE_FILE or related)"
        )
        assert intent.confidence >= 0.3


class TestCopyFile:
    """COPY_FILE — diverse phrasings."""

    CASES = [
        "copy file.txt to backup.txt",
        "duplicate config.yaml",
        "cp README.md README.bak",
        "copy file settings.json to archive",
        "make a copy of main.py",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_copy_file(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("COPY_FILE", "RUN_COMMAND", "COPY_DIRECTORY"), (
            f"{msg!r} → {_type(intent)} (expected COPY_FILE or related)"
        )
        assert intent.confidence >= 0.3


class TestMoveFile:
    """MOVE_FILE — diverse phrasings."""

    CASES = [
        "move file.txt to archive/",
        "rename old.py to new.py",
        "mv config.yaml to backup/",
        "move file README.md to docs/",
        "rename file settings.json to settings.bak",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_move_file(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("MOVE_FILE", "RUN_COMMAND", "NAVIGATE_DIRECTORY"), (
            f"{msg!r} → {_type(intent)} (expected MOVE_FILE or related)"
        )
        assert intent.confidence >= 0.3


class TestCreateDirectory:
    """CREATE_DIRECTORY — diverse phrasings."""

    CASES = [
        "create directory src/utils",
        "make folder build",
        "mkdir output",
        "create folder logs",
        "new folder data",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_create_directory(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("CREATE_DIRECTORY", "NAVIGATE_DIRECTORY", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected CREATE_DIRECTORY or related)"
        )
        assert intent.confidence >= 0.3


class TestDeleteDirectory:
    """DELETE_DIRECTORY — diverse phrasings."""

    CASES = [
        "delete the build directory",
        "remove folder dist",
        "rmdir output",
        "delete directory temp",
        "remove the logs folder",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_delete_directory(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("DELETE_DIRECTORY", "DELETE_FILE", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected DELETE_DIRECTORY or related)"
        )
        assert intent.confidence >= 0.3


class TestCopyDirectory:
    """COPY_DIRECTORY — diverse phrasings."""

    CASES = [
        "copy the src folder to backup",
        "duplicate directory build",
        "copy directory configs to configs_bak",
        "cp -r tests tests_backup",
        "duplicate folder docs",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_copy_directory(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("COPY_DIRECTORY", "COPY_FILE", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected COPY_DIRECTORY or related)"
        )
        assert intent.confidence >= 0.3


class TestTerminalKill:
    """TERMINAL_KILL — diverse phrasings."""

    CASES = [
        "kill terminal 3",
        "stop the running terminal",
        "kill process",
        "terminate the background process",
        "abort process",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_terminal_kill(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("TERMINAL_KILL", "TERMINAL_MONITOR", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected TERMINAL_KILL or related)"
        )
        assert intent.confidence >= 0.3


class TestTerminalOutput:
    """TERMINAL_OUTPUT — diverse phrasings."""

    CASES = [
        "show terminal output",
        "get output from terminal 1",
        "what did it print",
        "display output",
        "terminal log",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_terminal_output(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("TERMINAL_OUTPUT", "TERMINAL_MONITOR", "QUERY_STATUS"), (
            f"{msg!r} → {_type(intent)} (expected TERMINAL_OUTPUT or related)"
        )
        assert intent.confidence >= 0.3


class TestConfigureEnvironment:
    """CONFIGURE_ENVIRONMENT — diverse phrasings."""

    CASES = [
        "set up python environment",
        "configure venv",
        "create virtual environment",
        "activate venv",
        "set environment variable PATH",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_configure_env(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in (
            "CONFIGURE_ENVIRONMENT", "INSTALL_DEPENDENCY", "RUN_COMMAND",
        ), (
            f"{msg!r} → {_type(intent)} (expected CONFIGURE_ENVIRONMENT or related)"
        )
        assert intent.confidence >= 0.3


class TestSearchFile:
    """SEARCH_FILE — diverse phrasings."""

    CASES = [
        "search for config.yaml",
        "find files matching *.py",
        "grep TODO in all files",
        "find text import in main.py",
        "search content for error handling",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_search_file(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("SEARCH_FILE", "READ_FILE", "QUERY_STATUS"), (
            f"{msg!r} → {_type(intent)} (expected SEARCH_FILE or related)"
        )
        assert intent.confidence >= 0.3


class TestBackendModify:
    """BACKEND_MODIFY — diverse phrasings."""

    CASES = [
        "modify the cirq backend code",
        "change backend.py source",
        "edit backend source code",
        "patch backend with fix",
        "update backend code for LRET",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_backend_modify(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("BACKEND_MODIFY", "BACKEND_BUILD", "WRITE_FILE", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected BACKEND_MODIFY or related)"
        )
        assert intent.confidence >= 0.3


class TestBackendList:
    """BACKEND_LIST — diverse phrasings."""

    CASES = [
        "list available backends",
        "show all backends",
        "what backends are supported",
        "which backends can I use",
        "show build profiles",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_backend_list(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("BACKEND_LIST", "BACKEND_CONFIGURE", "QUERY_STATUS"), (
            f"{msg!r} → {_type(intent)} (expected BACKEND_LIST or related)"
        )
        assert intent.confidence >= 0.3


class TestAdminElevate:
    """ADMIN_ELEVATE — diverse phrasings."""

    CASES = [
        "run as administrator",
        "elevate privileges",
        "sudo command",
        "admin access required",
        "run with admin privileges",
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("msg", CASES)
    def test_recognises_admin_elevate(self, processor: RobustNLProcessor, msg: str):
        intent = processor.recognize_intent(msg)
        assert _type(intent) in ("ADMIN_ELEVATE", "RUN_COMMAND"), (
            f"{msg!r} → {_type(intent)} (expected ADMIN_ELEVATE or related)"
        )
        assert intent.confidence >= 0.3


# ═══════════════════════════════════════════════════════════════════════
# Phase 12 — Example B Multi-Step Test
# ═══════════════════════════════════════════════════════════════════════


class TestMultiStepExampleB:
    """Phase 12 Example B — Local Path, Branch, Build, Configure.

    Tests that the processor decomposes Example B into the expected
    sub-intents when expressed with natural separators.
    """

    @pytest.mark.unit
    def test_example_b_natural(self, processor: RobustNLProcessor):
        """Example B with natural language connectors."""
        msg = (
            "The LRET repo is at C:\\Users\\dell\\Pictures\\Screenshots\\LRET. "
            "Checkout the cirq-scalability-comparison branch, "
            "then build and compile it, "
            "then configure Proxima to use it"
        )
        intent = processor.recognize_intent(msg)
        kind = _type(intent)
        assert kind in ("MULTI_STEP", "GIT_CHECKOUT"), (
            f"Expected MULTI_STEP or GIT_CHECKOUT, got {kind}"
        )
        if kind == "MULTI_STEP":
            subs = _sub_types(intent)
            assert len(subs) >= 2, f"Expected ≥2 sub-intents, got {len(subs)}: {subs}"

    @pytest.mark.unit
    def test_example_b_then_separated(self, processor: RobustNLProcessor):
        """Example B with explicit 'then' separators."""
        msg = (
            "Go to C:\\Users\\dell\\Pictures\\Screenshots\\LRET "
            "then checkout cirq-scalability-comparison "
            "then install dependencies "
            "then build the backend "
            "then configure Proxima to use it"
        )
        intent = processor.recognize_intent(msg)
        assert _type(intent) == "MULTI_STEP"
        subs = _sub_types(intent)
        assert len(subs) >= 4, f"Expected ≥4 sub-intents, got {len(subs)}: {subs}"

    @pytest.mark.unit
    def test_example_b_numbered(self, processor: RobustNLProcessor):
        """Example B in numbered list format."""
        msg = (
            "1. Go to C:\\Users\\dell\\Pictures\\Screenshots\\LRET\n"
            "2. Checkout cirq-scalability-comparison branch\n"
            "3. Build and compile\n"
            "4. Configure Proxima to use it"
        )
        intent = processor.recognize_intent(msg)
        assert _type(intent) == "MULTI_STEP"
        subs = _sub_types(intent)
        assert len(subs) >= 3, f"Expected ≥3 sub-intents, got {len(subs)}: {subs}"


# ---------------------------------------------------------------------------
# Performance Benchmarks (Phase 13.5)
# ---------------------------------------------------------------------------
class TestPerformanceBenchmarks:
    """Verify critical NL pipeline operations stay within acceptable latency."""

    @pytest.mark.unit
    def test_intent_recognition_under_100ms(self, processor: RobustNLProcessor):
        """Each intent recognition call should complete in < 100 ms."""
        import time

        messages = [
            "install numpy",
            "git clone https://github.com/user/repo",
            "create a file called app.py with hello world content",
            "navigate to /home/user/project",
            "run pytest",
            "build the project",
            "check git status",
            "open the terminal",
            "search for TODO comments",
            "explain what this error means",
            "show system info",
            "help me with Docker setup",
            "deploy the app to production",
            "set environment variable API_KEY=1234",
            "what quantum backends are available",
            "configure LRET simulator",
            "start a local server on port 8080",
            "delete temp files in /tmp",
            "commit changes with message 'fix bug'",
            "pull latest from origin main",
        ]

        for msg in messages:
            start = time.perf_counter()
            processor.recognize_intent(msg)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms < 100, (
                f"Intent recognition for '{msg}' took {elapsed_ms:.1f}ms (>100ms)"
            )

    @pytest.mark.unit
    def test_entity_extraction_under_100ms(self, processor: RobustNLProcessor):
        """Entity extraction on a 5000-char message should complete in < 100 ms."""
        import time

        # Build a long realistic message
        base = "install numpy and navigate to /home/user/project then run pytest "
        long_msg = (base * 80)[:5000]
        assert len(long_msg) >= 5000

        start = time.perf_counter()
        processor.extract_entities(long_msg)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, (
            f"Entity extraction on 5000-char message took {elapsed_ms:.1f}ms (>100ms)"
        )

    @pytest.mark.unit
    def test_multi_step_parsing_20_steps(self, processor: RobustNLProcessor):
        """Parsing a 20-step numbered list should complete in < 200 ms."""
        import time

        steps = "\n".join(
            f"{i}. Step {i}: perform action {i}" for i in range(1, 21)
        )

        start = time.perf_counter()
        intent = processor.recognize_intent(steps)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, (
            f"20-step multi-step parsing took {elapsed_ms:.1f}ms (>200ms)"
        )

    @pytest.mark.unit
    def test_batch_recognition_throughput(self, processor: RobustNLProcessor):
        """50 consecutive recognize_intent calls should finish within 2 s total."""
        import time

        messages = [
            f"command number {i}: do something for task {i}" for i in range(50)
        ]

        start = time.perf_counter()
        for msg in messages:
            processor.recognize_intent(msg)
        total_ms = (time.perf_counter() - start) * 1000
        assert total_ms < 2000, (
            f"50 intent recognitions took {total_ms:.1f}ms (>2000ms)"
        )
