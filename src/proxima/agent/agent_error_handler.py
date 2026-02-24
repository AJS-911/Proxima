"""Agent-specific error classification and recovery strategies (Phase 11, Step 11.1).

Wraps the existing :class:`ErrorClassifier` from ``error_detection.py`` and adds
recovery-strategy logic tailored to *terminal output* (strings + exit codes)
rather than Python exceptions.

The handler maps raw terminal output to :class:`ErrorCategory` values and
returns a human-readable message together with an optional suggested fix
command.  It also implements automatic retry with exponential back-off for
transient failures (Step 11.3).
"""

from __future__ import annotations

import logging
import os
import re
import time
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
    from proxima.agent.dynamic_tools.error_detection import (
        ErrorClassifier,
        ErrorContext,
    )
    from proxima.agent.dependency_manager import ProjectDependencyManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-export ErrorCategory so callers only need one import
# ---------------------------------------------------------------------------

try:
    from proxima.agent.dynamic_tools.error_detection import ErrorCategory
except ImportError:  # pragma: no cover – defensive fallback
    from enum import Enum

    class ErrorCategory(Enum):  # type: ignore[no-redef]
        """Minimal stub when error_detection is unavailable."""

        FILESYSTEM = "filesystem"
        NETWORK = "network"
        AUTHENTICATION = "authentication"
        PERMISSION = "permission"
        RESOURCE = "resource"
        TIMEOUT = "timeout"
        VALIDATION = "validation"
        CONFIGURATION = "configuration"
        DEPENDENCY = "dependency"
        GIT = "git"
        GITHUB = "github"
        TERMINAL = "terminal"
        BUILD = "build"
        RUNTIME = "runtime"
        MEMORY = "memory"
        DISK = "disk"
        SYNTAX = "syntax"
        LOGIC = "logic"
        CONCURRENCY = "concurrency"
        UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Pre-compiled classification patterns  (order matters – first match wins)
# ---------------------------------------------------------------------------
# Each entry:  (compiled_regex,  ErrorCategory,  human_message_template,  fix_template | None)
# ``{0}`` in templates is replaced with the first regex capture group (if any).

_CLASSIFICATION_PATTERNS: List[
    Tuple[re.Pattern[str], ErrorCategory, str, Optional[str]]
] = []


def _p(
    pattern: str,
    category: ErrorCategory,
    message: str,
    fix: Optional[str] = None,
) -> None:
    """Append a compiled pattern entry to the module-level list."""
    _CLASSIFICATION_PATTERNS.append(
        (re.compile(pattern, re.IGNORECASE | re.DOTALL), category, message, fix)
    )


# ── Permission ────────────────────────────────────────────────────────
_p(r"Permission denied", ErrorCategory.PERMISSION,
   "Permission denied — you may need admin/elevated privileges.",
   None)
_p(r"Access is denied", ErrorCategory.PERMISSION,
   "Access denied — try running with administrator privileges.",
   None)
_p(r"Operation not permitted", ErrorCategory.PERMISSION,
   "Operation not permitted — check file or directory permissions.",
   None)

# ── Authentication ────────────────────────────────────────────────────
_p(r"Authentication failed", ErrorCategory.AUTHENTICATION,
   "Authentication failed — check your credentials.",
   None)
_p(r"Invalid credentials", ErrorCategory.AUTHENTICATION,
   "Invalid credentials — verify your API key or password.",
   None)
_p(r"could not read Username", ErrorCategory.AUTHENTICATION,
   "Git credential prompt failed — configure credential helper.",
   "git config --global credential.helper store")
_p(r"(?:invalid|missing|bad).*api.?key", ErrorCategory.AUTHENTICATION,
   "API key is invalid or missing.",
   None)
_p(r"SSH key.*(?:denied|invalid|not found)", ErrorCategory.AUTHENTICATION,
   "SSH key issue — check your SSH configuration.",
   None)

# ── Network ───────────────────────────────────────────────────────────
_p(r"Could not resolve host", ErrorCategory.NETWORK,
   "DNS resolution failed — check your internet connection.",
   None)
_p(r"Connection refused", ErrorCategory.NETWORK,
   "Connection refused — the remote server may be down.",
   None)
_p(r"(?:Connection|Read) timed out", ErrorCategory.NETWORK,
   "Network timeout — try again or check your connection.",
   None)
_p(r"SSL(?:Error|:)", ErrorCategory.NETWORK,
   "SSL error — certificate verification failed.",
   None)
_p(r"Network is unreachable", ErrorCategory.NETWORK,
   "Network unreachable — check your internet connection.",
   None)

# ── Dependency ────────────────────────────────────────────────────────
_p(r"ModuleNotFoundError.*No module named '([^']+)'", ErrorCategory.DEPENDENCY,
   "Python module '{0}' is not installed.",
   "pip install {0}")
_p(r"No module named '([^']+)'", ErrorCategory.DEPENDENCY,
   "Missing Python module '{0}'.",
   "pip install {0}")
_p(r"ImportError.*cannot import name '([^']+)'",
   ErrorCategory.DEPENDENCY,
   "Cannot import '{0}' — the package may need updating.",
   "pip install --upgrade {0}")
_p(r"ERROR: No matching distribution found for (\S+)",
   ErrorCategory.DEPENDENCY,
   "Package '{0}' not found on PyPI.",
   None)
_p(r"pkg_resources\.DistributionNotFound.*'([^']+)'",
   ErrorCategory.DEPENDENCY,
   "Distribution '{0}' not found.",
   "pip install {0}")

# ── Build ─────────────────────────────────────────────────────────────
_p(r"(?:cmake|CMake)\s+Error", ErrorCategory.BUILD,
   "CMake error — check the build configuration.",
   "pip install cmake")
_p(r"make\s*(?:\[\d+\])?\s*:\s*\*\*\*", ErrorCategory.BUILD,
   "Make build failed — check the compiler output.",
   None)
_p(r"error:.*(?:cl\.exe|gcc|g\+\+|clang).*(?:failed|not found)",
   ErrorCategory.BUILD,
   "C/C++ compiler error — build tools may be missing.",
   None)
_p(r"(?:compilation|compile)\s+(?:error|failed)", ErrorCategory.BUILD,
   "Compilation failed — review the error output above.",
   None)
_p(r"error C\d{4}:", ErrorCategory.BUILD,
   "MSVC compilation error.",
   None)

# ── Syntax ────────────────────────────────────────────────────────────
_p(r"SyntaxError", ErrorCategory.SYNTAX,
   "Python syntax error — check the affected file.",
   None)
_p(r"IndentationError", ErrorCategory.SYNTAX,
   "Python indentation error.",
   None)
_p(r"TabError", ErrorCategory.SYNTAX,
   "Mixed tabs and spaces.",
   None)

# ── Git ───────────────────────────────────────────────────────────────
_p(r"CONFLICT\s+\(", ErrorCategory.GIT,
   "Merge conflict detected — resolve the conflicting files.",
   None)
_p(r"fatal:\s+", ErrorCategory.GIT,
   "Git fatal error.",
   None)
_p(r"error:\s+Your local changes", ErrorCategory.GIT,
   "Dirty working tree — stash or commit changes first.",
   "git stash")
_p(r"error:\s+pathspec '([^']+)' did not match",
   ErrorCategory.GIT,
   "Branch or path '{0}' not found.",
   None)
_p(r"detached HEAD", ErrorCategory.GIT,
   "HEAD is detached.",
   None)

# ── Filesystem ────────────────────────────────────────────────────────
_p(r"No such file or directory.*'?([^'\":\n]+)'?",
   ErrorCategory.FILESYSTEM,
   "File or directory '{0}' not found.",
   None)
_p(r"not recognized as .*(?:command|program)",
   ErrorCategory.FILESYSTEM,
   "Command not recognised — check spelling or install the tool.",
   None)
_p(r"(?:File|Directory) not found", ErrorCategory.FILESYSTEM,
   "File or directory not found.",
   None)
_p(r"command not found", ErrorCategory.FILESYSTEM,
   "Command not found — verify the executable is on PATH.",
   None)

# ── Timeout ───────────────────────────────────────────────────────────
_p(r"(?:Timed out|timed out|TIMEOUT)", ErrorCategory.TIMEOUT,
   "Operation timed out — try with a longer timeout or run in background.",
   None)

# ── Resource / Memory / Disk ──────────────────────────────────────────
_p(r"No space left on device", ErrorCategory.DISK,
   "Disk is full — free up space and try again.",
   None)
_p(r"MemoryError", ErrorCategory.MEMORY,
   "Out of memory — close other applications or reduce workload.",
   None)
_p(r"(?:CUDA|GPU).*(?:out of memory|not available|error)",
   ErrorCategory.RESOURCE,
   "GPU resource error — check CUDA installation and available memory.",
   None)
_p(r"ResourceWarning|ResourceError", ErrorCategory.RESOURCE,
   "System resource issue.",
   None)

# ── Validation ────────────────────────────────────────────────────────
_p(r"(?:ValidationError|Invalid input|schema.*(?:mismatch|invalid))",
   ErrorCategory.VALIDATION,
   "Validation error — check the input format.",
   None)

# ── Configuration ─────────────────────────────────────────────────────
_p(r"(?:Config|Configuration).*(?:not found|missing|invalid)",
   ErrorCategory.CONFIGURATION,
   "Configuration error — the config file may be missing or invalid.",
   None)

# ── Terminal ──────────────────────────────────────────────────────────
_p(r"(?:shell|bash|powershell|cmd).*(?:error|crash|not found)",
   ErrorCategory.TERMINAL,
   "Shell error — verify the shell is available.",
   None)

# ── Runtime (generic) ─────────────────────────────────────────────────
_p(r"Traceback \(most recent call last\)", ErrorCategory.RUNTIME,
   "Python runtime error — see traceback above.",
   None)
_p(r"AssertionError", ErrorCategory.RUNTIME,
   "Assertion failed.",
   None)
_p(r"RuntimeError", ErrorCategory.RUNTIME,
   "Runtime error.",
   None)

# ── Logic / Concurrency ──────────────────────────────────────────────
_p(r"(?:deadlock|DeadlockError)", ErrorCategory.CONCURRENCY,
   "Deadlock detected.",
   None)
_p(r"(?:race condition|ThreadError|BrokenPipeError)",
   ErrorCategory.CONCURRENCY,
   "Concurrency issue.",
   None)


# ---------------------------------------------------------------------------
# Recovery strategies per category
# ---------------------------------------------------------------------------

_RECOVERY_STRATEGIES: Dict[ErrorCategory, str] = {
    ErrorCategory.PERMISSION:
        "Try running the command with elevated privileges or fix file permissions.",
    ErrorCategory.AUTHENTICATION:
        "Check your credentials, API keys, or SSH configuration.",
    ErrorCategory.NETWORK:
        "Check your internet connection and retry. If the problem persists, try a longer timeout.",
    ErrorCategory.DEPENDENCY:
        "Install the missing dependency (see suggested fix above).",
    ErrorCategory.BUILD:
        "Install required build tools (CMake, compiler). Re-run with --verbose for details.",
    ErrorCategory.SYNTAX:
        "Fix the syntax error in the affected file. If caused by a recent modification, consider undoing it.",
    ErrorCategory.TIMEOUT:
        "Retry with a longer timeout or run the command in background.",
    ErrorCategory.RESOURCE:
        "Free up system resources (close applications, clear temp files) and retry.",
    ErrorCategory.MEMORY:
        "Reduce workload or close memory-intensive applications.",
    ErrorCategory.DISK:
        "Free up disk space — delete temporary files, build caches, or unused dependencies.",
    ErrorCategory.GIT:
        "Resolve the git issue: stash changes, fix conflicts, or reset state.",
    ErrorCategory.FILESYSTEM:
        "Verify the file/directory exists and the path is correct.",
    ErrorCategory.RUNTIME:
        "Review the traceback for clues. Debug the affected code.",
    ErrorCategory.VALIDATION:
        "Check the input against the expected format.",
    ErrorCategory.CONFIGURATION:
        "Create or fix the configuration file.",
    ErrorCategory.TERMINAL:
        "Verify the shell is installed and on PATH.",
    ErrorCategory.LOGIC:
        "Review the code logic. Consider adding debug logging.",
    ErrorCategory.CONCURRENCY:
        "Review thread synchronisation. Add locks or simplify concurrency.",
    ErrorCategory.GITHUB:
        "Check GitHub API rate limits, tokens, or repository access.",
    ErrorCategory.UNKNOWN:
        "An unexpected error occurred. Review the error output for details.",
}


# ---------------------------------------------------------------------------
# Retry configuration per category  (max_retries, base_delay_seconds)
# ---------------------------------------------------------------------------

_RETRY_CONFIG: Dict[ErrorCategory, Tuple[int, float]] = {
    ErrorCategory.NETWORK:      (3, 2.0),   # 2s → 4s → 8s
    ErrorCategory.TIMEOUT:      (1, 0.0),   # retry once with doubled timeout
    ErrorCategory.DEPENDENCY:   (1, 1.0),   # retry once after suggested fix
    ErrorCategory.BUILD:        (1, 0.5),   # retry once with verbose
    ErrorCategory.GITHUB:       (2, 3.0),   # rate-limit back-off
    ErrorCategory.AUTHENTICATION: (0, 0.0), # no auto-retry
    ErrorCategory.PERMISSION:   (0, 0.0),   # no auto-retry
    ErrorCategory.FILESYSTEM:   (0, 0.0),   # deterministic
    ErrorCategory.SYNTAX:       (0, 0.0),   # deterministic
    ErrorCategory.VALIDATION:   (0, 0.0),   # deterministic
}
# Everything else defaults to (0, 0.0) — no auto-retry.

_DEFAULT_RETRY = (0, 0.0)


# ---------------------------------------------------------------------------
# Risk levels for fix commands
# ---------------------------------------------------------------------------

_FIX_RISK_LEVELS: Dict[ErrorCategory, str] = {
    ErrorCategory.DEPENDENCY: "low",
    ErrorCategory.BUILD:      "low",
    ErrorCategory.NETWORK:    "low",
    ErrorCategory.GIT:        "medium",
    ErrorCategory.FILESYSTEM: "medium",
    ErrorCategory.PERMISSION: "high",
    ErrorCategory.CONFIGURATION: "low",
}
_DEFAULT_FIX_RISK = "medium"


# ═══════════════════════════════════════════════════════════════════════
# AgentErrorHandler
# ═══════════════════════════════════════════════════════════════════════

class AgentErrorHandler:
    """Classifies terminal output errors and proposes recovery strategies.

    This class **wraps** the existing :class:`ErrorClassifier` from
    ``error_detection.py`` for exception-based classification, while
    adding its own regex pipeline for classifying *raw terminal output*
    (strings + exit codes).

    Parameters
    ----------
    error_classifier : ErrorClassifier or None
        The existing classifier, used as a secondary classification source
        when the terminal-output regex pipeline finds no match.
    dep_manager : ProjectDependencyManager or None
        Used to enrich fix suggestions for dependency errors.
    """

    def __init__(
        self,
        error_classifier: Optional["ErrorClassifier"] = None,
        dep_manager: Optional["ProjectDependencyManager"] = None,
    ) -> None:
        self._classifier = error_classifier
        self._dep_manager = dep_manager

        # ── Logging setup ─────────────────────────────────────────────
        self._setup_logging()

    # ── Public API ────────────────────────────────────────────────────

    def classify_output(
        self,
        error_output: str,
        exit_code: int = 1,
    ) -> Tuple[ErrorCategory, str, Optional[str]]:
        """Classify raw terminal output into an error category.

        Parameters
        ----------
        error_output:
            Raw stderr / combined stdout+stderr text from a failed command.
        exit_code:
            Process exit code (0 = success, 124 commonly = timeout).

        Returns
        -------
        tuple
            ``(category, human_readable_message, suggested_fix_command)``
            where *suggested_fix_command* may be ``None``.
        """
        if not error_output and exit_code == 0:
            return (ErrorCategory.UNKNOWN, "No error output captured.", None)

        # Special exit codes
        if exit_code == 124:
            return (
                ErrorCategory.TIMEOUT,
                "Command timed out (exit code 124).",
                None,
            )

        # Run through pattern list (first match wins)
        for pattern, category, msg_template, fix_template in _CLASSIFICATION_PATTERNS:
            m = pattern.search(error_output)
            if m:
                # Fill template with first capture group
                captured = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
                captured = self._normalise_module_name(captured)
                message = msg_template.format(captured) if "{0}" in msg_template else msg_template
                fix = fix_template.format(captured) if fix_template and "{0}" in fix_template else fix_template

                # Enrich dependency fixes via ProjectDependencyManager
                if category == ErrorCategory.DEPENDENCY and self._dep_manager is not None:
                    enriched = self._try_dep_manager_fix(error_output)
                    if enriched:
                        fix = enriched

                return (category, message, fix)

        # Fallback: try the existing ErrorClassifier on a synthetic exception
        if self._classifier is not None:
            fallback = self._try_classifier_fallback(error_output, exit_code)
            if fallback is not None:
                return fallback

        return (
            ErrorCategory.UNKNOWN,
            f"Unrecognised error (exit code {exit_code}).",
            None,
        )

    def get_recovery_strategy(self, category: ErrorCategory) -> str:
        """Return a human-readable recovery strategy for *category*."""
        return _RECOVERY_STRATEGIES.get(category, _RECOVERY_STRATEGIES[ErrorCategory.UNKNOWN])

    def get_retry_config(self, category: ErrorCategory) -> Tuple[int, float]:
        """Return ``(max_retries, base_delay_seconds)`` for *category*.

        Delay is doubled on each subsequent retry (exponential back-off).
        """
        return _RETRY_CONFIG.get(category, _DEFAULT_RETRY)

    def get_fix_risk_level(self, category: ErrorCategory) -> str:
        """Return the risk level for auto-applying a fix: low / medium / high."""
        return _FIX_RISK_LEVELS.get(category, _DEFAULT_FIX_RISK)

    def format_error_report(
        self,
        operation: str,
        error_output: str,
        exit_code: int = 1,
    ) -> Tuple[str, ErrorCategory, Optional[str]]:
        """Classify an error and return a formatted user-facing report.

        Returns
        -------
        tuple
            ``(formatted_message, category, suggested_fix)``
        """
        category, message, fix = self.classify_output(error_output, exit_code)
        recovery = self.get_recovery_strategy(category)

        parts = [f"❌ **{operation}** failed: {message}"]
        details = error_output.strip()[:500]
        if details:
            parts.append(f"\n📋 **Details:** {details}")
        parts.append(f"\n💡 **Suggested fix:** {fix or 'No automatic fix available.'}")
        parts.append(f"\n🔧 **Recovery:** {recovery}")

        return ("\n".join(parts), category, fix)

    def should_auto_retry(
        self,
        category: ErrorCategory,
        attempt: int,
    ) -> Tuple[bool, float]:
        """Determine whether an automatic retry should be attempted.

        Parameters
        ----------
        category:
            The classified error category.
        attempt:
            Current attempt number (0-based: 0 = first failure).

        Returns
        -------
        tuple
            ``(should_retry: bool, delay_seconds: float)``
        """
        max_retries, base_delay = self.get_retry_config(category)
        if attempt >= max_retries:
            return (False, 0.0)
        delay = base_delay * (2 ** attempt)
        return (True, delay)

    # ── Private helpers ───────────────────────────────────────────────

    def _try_dep_manager_fix(self, error_output: str) -> Optional[str]:
        """Ask ProjectDependencyManager for a richer fix suggestion."""
        if self._dep_manager is None:
            return None
        try:
            return self._dep_manager.detect_and_fix_errors(error_output, os.getcwd())
        except Exception:
            return None

    def _try_classifier_fallback(
        self,
        error_output: str,
        exit_code: int,
    ) -> Optional[Tuple[ErrorCategory, str, Optional[str]]]:
        """Use the existing ErrorClassifier as a fallback."""
        try:
            exc = RuntimeError(error_output[:500])
            from proxima.agent.dynamic_tools.error_detection import ErrorContext
            from datetime import datetime

            ctx = ErrorContext(
                timestamp=datetime.now(),
                operation="terminal_command",
                working_directory=os.getcwd(),
            )
            cat, _sev, _pat = self._classifier.classify(exc, ctx)  # type: ignore[union-attr]
            return (cat, f"Error classified as {cat.value}.", None)
        except Exception:
            return None

    @staticmethod
    def _normalise_module_name(name: str) -> str:
        """Map common import names to PyPI package names."""
        _MAP = {
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "sklearn": "scikit-learn",
            "yaml": "pyyaml",
            "bs4": "beautifulsoup4",
            "attr": "attrs",
            "gi": "pygobject",
            "Crypto": "pycryptodome",
            "serial": "pyserial",
            "usb": "pyusb",
            "wx": "wxPython",
        }
        return _MAP.get(name, name)

    def _setup_logging(self) -> None:
        """Ensure agent error logs go to ``~/.proxima/logs/agent.log``."""
        try:
            log_dir = os.path.join(os.path.expanduser("~"), ".proxima", "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "agent.log")

            # Only add handler if not already present
            for h in logger.handlers:
                if isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path):
                    return

            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(fh)
        except Exception:
            pass  # non-critical — logging setup should never crash the agent


# ═══════════════════════════════════════════════════════════════════════
# Retry executor  (Step 11.3)
# ═══════════════════════════════════════════════════════════════════════

class RetryExecutor:
    """Executes a callable with category-aware retry logic.

    Integrates with :class:`AgentErrorHandler` to determine whether and
    how long to wait before each retry, and to optionally apply a fix
    command between attempts.

    Parameters
    ----------
    error_handler : AgentErrorHandler
        Used for classification and retry-config look-up.
    ui_callback : callable or None
        ``(message: str) -> None`` — sends status updates to the TUI.
    execute_fix : callable or None
        ``(fix_command: str) -> str`` — runs a fix command, returns output.
    """

    def __init__(
        self,
        error_handler: AgentErrorHandler,
        ui_callback: Optional[Callable[[str], None]] = None,
        execute_fix: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._handler = error_handler
        self._ui = ui_callback or (lambda _s: None)
        self._execute_fix = execute_fix

    def execute_with_retry(
        self,
        operation_name: str,
        callable_fn: Callable[[], Any],
        error_extractor: Optional[Callable[[Any], Tuple[str, int]]] = None,
    ) -> Any:
        """Run *callable_fn*, retrying on classified transient failures.

        Parameters
        ----------
        operation_name:
            Human-readable name for logging / UI messages.
        callable_fn:
            Zero-argument callable that performs the operation and returns
            a result.  Should raise or return a failure indicator.
        error_extractor:
            Given a result, return ``(error_output, exit_code)`` if the
            result indicates failure, or ``("", 0)`` if success.

        Returns
        -------
        The result from the last invocation of *callable_fn*.
        """
        attempt = 0
        last_category: Optional[ErrorCategory] = None
        last_fix: Optional[str] = None

        while True:
            try:
                result = callable_fn()
            except Exception as exc:
                error_text = str(exc)
                exit_code = getattr(exc, "returncode", 1)
                category, message, fix = self._handler.classify_output(error_text, exit_code)
                logger.exception(
                    "Operation '%s' attempt %d failed [%s]: %s",
                    operation_name, attempt + 1, category.value, message,
                )
                last_category = category
                last_fix = fix

                should, delay = self._handler.should_auto_retry(category, attempt)
                if should:
                    self._apply_retry(attempt, delay, fix, category)
                    attempt += 1
                    continue
                raise

            # Check result for failure (non-exception path)
            if error_extractor is not None:
                error_text, exit_code = error_extractor(result)
                if exit_code != 0 and error_text:
                    category, message, fix = self._handler.classify_output(error_text, exit_code)
                    logger.warning(
                        "Operation '%s' attempt %d returned error [%s]: %s",
                        operation_name, attempt + 1, category.value, message,
                    )
                    last_category = category
                    last_fix = fix

                    should, delay = self._handler.should_auto_retry(category, attempt)
                    if should:
                        self._apply_retry(attempt, delay, fix, category)
                        attempt += 1
                        continue

            return result

    # ── helpers ───────────────────────────────────────────────────────

    def _apply_retry(
        self,
        attempt: int,
        delay: float,
        fix: Optional[str],
        category: ErrorCategory,
    ) -> None:
        """Sleep, optionally apply fix, and inform the UI."""
        self._ui(f"🔄 Retry {attempt + 1}...")

        # Apply suggested fix before retrying (if available and low risk)
        if fix and self._execute_fix is not None:
            risk = self._handler.get_fix_risk_level(category)
            if risk == "low":
                self._ui(f"🔧 Applying fix: `{fix}`")
                try:
                    self._execute_fix(fix)
                except Exception as fix_exc:
                    logger.warning("Fix command failed: %s", fix_exc)

        # Exponential back-off delay
        if delay > 0:
            time.sleep(delay)
