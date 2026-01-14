"""Enhanced State Machine Implementation for Execution Flow.

Implements the execution lifecycle with explicit states and transitions,
plus critical missing features:
- State persistence during failures
- Resource cleanup on abort
- Complex transition validation

The state machine is agnostic to whether planning and execution are
driven by LLM-backed planners or local models - it only manages
lifecycle and visibility.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, TypeVar

from transitions import Machine

from proxima.utils.logging import get_logger


# ==================== EXECUTION STATES ====================


class ExecutionState(str, Enum):
    """Execution lifecycle states."""

    IDLE = "IDLE"
    PLANNING = "PLANNING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"
    ERROR = "ERROR"
    RECOVERING = "RECOVERING"  # New: recovering from failure


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(
        self,
        message: str,
        from_state: str,
        to_state: str,
        trigger: str,
    ):
        super().__init__(message)
        self.from_state = from_state
        self.to_state = to_state
        self.trigger = trigger


class StatePersistenceError(Exception):
    """Raised when state persistence fails."""

    pass


# ==================== STATE PERSISTENCE ====================


@dataclass
class PersistedState:
    """Represents persisted state data."""

    execution_id: str
    state: str
    history: list[str]
    context_data: dict[str, Any]
    resources: list[str]
    timestamp: float
    error_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "execution_id": self.execution_id,
            "state": self.state,
            "history": self.history,
            "context_data": self.context_data,
            "resources": self.resources,
            "timestamp": self.timestamp,
            "error_info": self.error_info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersistedState:
        """Create from dictionary."""
        return cls(
            execution_id=data["execution_id"],
            state=data["state"],
            history=data["history"],
            context_data=data.get("context_data", {}),
            resources=data.get("resources", []),
            timestamp=data["timestamp"],
            error_info=data.get("error_info"),
        )


class StatePersistence:
    """Handles state persistence during failures.

    Features:
    - Periodic state snapshots
    - Crash recovery from persisted state
    - Atomic writes to prevent corruption
    """

    def __init__(
        self,
        storage_dir: Path | None = None,
        auto_persist_interval: float = 5.0,
    ):
        """Initialize state persistence.

        Args:
            storage_dir: Directory for state files
            auto_persist_interval: Seconds between auto-persists
        """
        self.storage_dir = storage_dir or Path(tempfile.gettempdir()) / "proxima_state"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.auto_persist_interval = auto_persist_interval
        self.logger = get_logger("state.persistence")

        self._persist_lock = threading.Lock()
        self._auto_persist_thread: threading.Thread | None = None
        self._auto_persist_stop = threading.Event()

    def persist(self, state_data: PersistedState) -> Path:
        """Persist state to disk atomically.

        Args:
            state_data: State data to persist

        Returns:
            Path to the persisted state file
        """
        with self._persist_lock:
            state_file = self.storage_dir / f"{state_data.execution_id}.state.json"
            temp_file = state_file.with_suffix(".tmp")

            try:
                # Write to temp file first
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(state_data.to_dict(), f, indent=2)

                # Atomic rename
                temp_file.replace(state_file)

                self.logger.debug(
                    "state.persisted",
                    execution_id=state_data.execution_id,
                    state=state_data.state,
                )
                return state_file

            except Exception as exc:
                if temp_file.exists():
                    temp_file.unlink()
                raise StatePersistenceError(
                    f"Failed to persist state: {exc}"
                ) from exc

    def load(self, execution_id: str) -> PersistedState | None:
        """Load persisted state from disk.

        Args:
            execution_id: Execution ID to load

        Returns:
            Persisted state or None if not found
        """
        state_file = self.storage_dir / f"{execution_id}.state.json"

        if not state_file.exists():
            return None

        try:
            with open(state_file, encoding="utf-8") as f:
                data = json.load(f)
            return PersistedState.from_dict(data)
        except Exception as exc:
            self.logger.warning(
                "state.load_failed",
                execution_id=execution_id,
                error=str(exc),
            )
            return None

    def delete(self, execution_id: str) -> bool:
        """Delete persisted state.

        Args:
            execution_id: Execution ID to delete

        Returns:
            True if deleted
        """
        state_file = self.storage_dir / f"{execution_id}.state.json"
        if state_file.exists():
            state_file.unlink()
            return True
        return False

    def list_persisted(self) -> list[str]:
        """List all persisted execution IDs."""
        return [
            f.stem.replace(".state", "")
            for f in self.storage_dir.glob("*.state.json")
        ]

    def recover_crashed_states(self) -> list[PersistedState]:
        """Find states that need recovery (crashed mid-execution).

        Returns:
            List of states that were running when interrupted
        """
        crashed = []
        for exec_id in self.list_persisted():
            state = self.load(exec_id)
            if state and state.state in (
                ExecutionState.RUNNING.value,
                ExecutionState.PAUSED.value,
                ExecutionState.PLANNING.value,
            ):
                crashed.append(state)
        return crashed

    def start_auto_persist(
        self,
        state_provider: Callable[[], PersistedState | None],
    ) -> None:
        """Start automatic periodic persistence.

        Args:
            state_provider: Callable that returns current state to persist
        """
        if self._auto_persist_thread is not None:
            return

        self._auto_persist_stop.clear()

        def persist_loop():
            while not self._auto_persist_stop.wait(self.auto_persist_interval):
                try:
                    state = state_provider()
                    if state:
                        self.persist(state)
                except Exception as exc:
                    self.logger.warning("auto_persist_failed", error=str(exc))

        self._auto_persist_thread = threading.Thread(
            target=persist_loop,
            daemon=True,
            name="state-auto-persist",
        )
        self._auto_persist_thread.start()

    def stop_auto_persist(self) -> None:
        """Stop automatic persistence."""
        if self._auto_persist_thread is None:
            return
        self._auto_persist_stop.set()
        self._auto_persist_thread.join(timeout=2.0)
        self._auto_persist_thread = None


# ==================== RESOURCE CLEANUP ====================


class ResourceHandle(ABC):
    """Abstract base for managed resources that need cleanup."""

    @property
    @abstractmethod
    def resource_id(self) -> str:
        """Unique identifier for this resource."""
        pass

    @property
    @abstractmethod
    def resource_type(self) -> str:
        """Type of resource (e.g., 'file', 'memory', 'connection')."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release/cleanup this resource."""
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Check if resource is still active."""
        pass


@dataclass
class FileResource(ResourceHandle):
    """File-based resource handle."""

    path: Path
    _id: str = field(default_factory=lambda: str(time.time()))
    _active: bool = True

    @property
    def resource_id(self) -> str:
        return self._id

    @property
    def resource_type(self) -> str:
        return "file"

    def cleanup(self) -> None:
        """Delete the file if it exists."""
        if self.path.exists():
            try:
                self.path.unlink()
            except Exception:
                pass
        self._active = False

    def is_active(self) -> bool:
        return self._active and self.path.exists()


@dataclass
class MemoryResource(ResourceHandle):
    """Memory-based resource handle."""

    data: Any
    size_bytes: int
    _id: str = field(default_factory=lambda: str(time.time()))
    _active: bool = True

    @property
    def resource_id(self) -> str:
        return self._id

    @property
    def resource_type(self) -> str:
        return "memory"

    def cleanup(self) -> None:
        """Release memory reference."""
        self.data = None
        self._active = False

    def is_active(self) -> bool:
        return self._active and self.data is not None


class ResourceCleanupManager:
    """Manages resource cleanup on abort or error.

    Features:
    - Track registered resources
    - Cleanup in reverse order of registration
    - Handle cleanup errors gracefully
    - Support resource grouping
    """

    def __init__(self):
        """Initialize the resource cleanup manager."""
        self.logger = get_logger("state.cleanup")
        self._resources: dict[str, ResourceHandle] = {}
        self._resource_groups: dict[str, list[str]] = {}
        self._cleanup_callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    def register(
        self,
        resource: ResourceHandle,
        group: str | None = None,
    ) -> str:
        """Register a resource for cleanup.

        Args:
            resource: Resource to track
            group: Optional group name for batch cleanup

        Returns:
            Resource ID
        """
        with self._lock:
            self._resources[resource.resource_id] = resource

            if group:
                if group not in self._resource_groups:
                    self._resource_groups[group] = []
                self._resource_groups[group].append(resource.resource_id)

            self.logger.debug(
                "resource.registered",
                resource_id=resource.resource_id,
                resource_type=resource.resource_type,
                group=group,
            )
            return resource.resource_id

    def unregister(self, resource_id: str) -> bool:
        """Unregister a resource (already cleaned up manually).

        Args:
            resource_id: Resource ID to unregister

        Returns:
            True if found and unregistered
        """
        with self._lock:
            if resource_id in self._resources:
                del self._resources[resource_id]

                # Remove from groups
                for group_resources in self._resource_groups.values():
                    if resource_id in group_resources:
                        group_resources.remove(resource_id)

                return True
            return False

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add a cleanup callback to be called on abort.

        Args:
            callback: Function to call during cleanup
        """
        with self._lock:
            self._cleanup_callbacks.append(callback)

    def cleanup_resource(self, resource_id: str) -> bool:
        """Cleanup a specific resource.

        Args:
            resource_id: Resource to cleanup

        Returns:
            True if successfully cleaned up
        """
        with self._lock:
            resource = self._resources.get(resource_id)
            if not resource:
                return False

            try:
                resource.cleanup()
                self.logger.debug(
                    "resource.cleaned",
                    resource_id=resource_id,
                    resource_type=resource.resource_type,
                )
            except Exception as exc:
                self.logger.warning(
                    "resource.cleanup_failed",
                    resource_id=resource_id,
                    error=str(exc),
                )
            finally:
                del self._resources[resource_id]

            return True

    def cleanup_group(self, group: str) -> int:
        """Cleanup all resources in a group.

        Args:
            group: Group name

        Returns:
            Number of resources cleaned up
        """
        with self._lock:
            if group not in self._resource_groups:
                return 0

            resource_ids = list(self._resource_groups[group])
            count = 0

            for resource_id in resource_ids:
                if self.cleanup_resource(resource_id):
                    count += 1

            del self._resource_groups[group]
            return count

    def cleanup_all(self) -> dict[str, int]:
        """Cleanup all resources and run callbacks.

        Returns:
            Dict with cleanup statistics
        """
        stats = {"resources": 0, "callbacks": 0, "errors": 0}

        with self._lock:
            # Cleanup resources in reverse order
            resource_ids = list(reversed(list(self._resources.keys())))

            for resource_id in resource_ids:
                try:
                    resource = self._resources.get(resource_id)
                    if resource:
                        resource.cleanup()
                        stats["resources"] += 1
                except Exception as exc:
                    self.logger.warning(
                        "resource.cleanup_all_failed",
                        resource_id=resource_id,
                        error=str(exc),
                    )
                    stats["errors"] += 1

            self._resources.clear()
            self._resource_groups.clear()

            # Run cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                    stats["callbacks"] += 1
                except Exception as exc:
                    self.logger.warning("callback.cleanup_failed", error=str(exc))
                    stats["errors"] += 1

            self._cleanup_callbacks.clear()

        self.logger.info("cleanup.completed", **stats)
        return stats

    def get_active_resources(self) -> list[dict[str, Any]]:
        """Get list of active resources."""
        with self._lock:
            return [
                {
                    "id": r.resource_id,
                    "type": r.resource_type,
                    "active": r.is_active(),
                }
                for r in self._resources.values()
            ]


# ==================== COMPLEX TRANSITION VALIDATION ====================


@dataclass
class TransitionRule:
    """Defines a transition rule with conditions."""

    trigger: str
    source: str | list[str]
    dest: str
    conditions: list[Callable[[dict[str, Any]], bool]] = field(default_factory=list)
    before_callbacks: list[Callable[[dict[str, Any]], None]] = field(
        default_factory=list
    )
    after_callbacks: list[Callable[[dict[str, Any]], None]] = field(
        default_factory=list
    )
    validators: list[Callable[[dict[str, Any]], tuple[bool, str]]] = field(
        default_factory=list
    )
    description: str = ""


class TransitionValidator:
    """Complex transition validation with custom rules.

    Features:
    - Pre-transition validation
    - Conditional transitions
    - Before/after callbacks
    - Transition history with reasons
    """

    def __init__(self):
        """Initialize the transition validator."""
        self.logger = get_logger("state.validator")
        self._rules: dict[str, list[TransitionRule]] = {}
        self._transition_history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_rule(self, rule: TransitionRule) -> None:
        """Add a transition rule.

        Args:
            rule: Transition rule to add
        """
        with self._lock:
            if rule.trigger not in self._rules:
                self._rules[rule.trigger] = []
            self._rules[rule.trigger].append(rule)

    def remove_rule(self, trigger: str, source: str | None = None) -> int:
        """Remove transition rules.

        Args:
            trigger: Trigger name
            source: Optional source state filter

        Returns:
            Number of rules removed
        """
        with self._lock:
            if trigger not in self._rules:
                return 0

            if source is None:
                count = len(self._rules[trigger])
                del self._rules[trigger]
                return count

            before = len(self._rules[trigger])
            self._rules[trigger] = [
                r
                for r in self._rules[trigger]
                if not (
                    r.source == source
                    or (isinstance(r.source, list) and source in r.source)
                )
            ]
            return before - len(self._rules[trigger])

    def validate_transition(
        self,
        trigger: str,
        current_state: str,
        context: dict[str, Any],
    ) -> tuple[bool, str, TransitionRule | None]:
        """Validate if a transition is allowed.

        Args:
            trigger: Transition trigger
            current_state: Current state
            context: Execution context

        Returns:
            Tuple of (is_valid, reason, matching_rule)
        """
        with self._lock:
            rules = self._rules.get(trigger, [])

            for rule in rules:
                # Check source state match
                sources = (
                    rule.source if isinstance(rule.source, list) else [rule.source]
                )
                if current_state not in sources and "*" not in sources:
                    continue

                # Check conditions
                conditions_met = all(cond(context) for cond in rule.conditions)
                if not conditions_met:
                    continue

                # Run validators
                for validator in rule.validators:
                    is_valid, reason = validator(context)
                    if not is_valid:
                        return False, reason, rule

                return True, "Transition allowed", rule

            return False, f"No matching rule for trigger '{trigger}' from state '{current_state}'", None

    def execute_transition(
        self,
        trigger: str,
        current_state: str,
        context: dict[str, Any],
    ) -> tuple[bool, str, str | None]:
        """Execute a validated transition with callbacks.

        Args:
            trigger: Transition trigger
            current_state: Current state
            context: Execution context

        Returns:
            Tuple of (success, reason, new_state)
        """
        is_valid, reason, rule = self.validate_transition(
            trigger, current_state, context
        )

        if not is_valid or rule is None:
            return False, reason, None

        # Execute before callbacks
        for callback in rule.before_callbacks:
            try:
                callback(context)
            except Exception as exc:
                return False, f"Before callback failed: {exc}", None

        # Record transition
        with self._lock:
            self._transition_history.append(
                {
                    "trigger": trigger,
                    "from_state": current_state,
                    "to_state": rule.dest,
                    "timestamp": time.time(),
                    "reason": reason,
                }
            )

        # Execute after callbacks
        for callback in rule.after_callbacks:
            try:
                callback(context)
            except Exception as exc:
                self.logger.warning(
                    "after_callback_failed",
                    trigger=trigger,
                    error=str(exc),
                )

        return True, reason, rule.dest

    def get_transition_history(self) -> list[dict[str, Any]]:
        """Get transition history."""
        with self._lock:
            return list(self._transition_history)

    def clear_history(self) -> None:
        """Clear transition history."""
        with self._lock:
            self._transition_history.clear()


# ==================== STANDARD TRANSITIONS ====================


TRANSITIONS: list[dict[str, Any]] = [
    {
        "trigger": "start",
        "source": ExecutionState.IDLE,
        "dest": ExecutionState.PLANNING,
    },
    {
        "trigger": "plan_complete",
        "source": ExecutionState.PLANNING,
        "dest": ExecutionState.READY,
    },
    {
        "trigger": "plan_failed",
        "source": ExecutionState.PLANNING,
        "dest": ExecutionState.ERROR,
    },
    {
        "trigger": "execute",
        "source": ExecutionState.READY,
        "dest": ExecutionState.RUNNING,
    },
    {
        "trigger": "pause",
        "source": ExecutionState.RUNNING,
        "dest": ExecutionState.PAUSED,
    },
    {
        "trigger": "resume",
        "source": ExecutionState.PAUSED,
        "dest": ExecutionState.RUNNING,
    },
    {
        "trigger": "complete",
        "source": ExecutionState.RUNNING,
        "dest": ExecutionState.COMPLETED,
    },
    {
        "trigger": "abort",
        "source": [ExecutionState.RUNNING, ExecutionState.PAUSED],
        "dest": ExecutionState.ABORTED,
    },
    {
        "trigger": "error",
        "source": ExecutionState.RUNNING,
        "dest": ExecutionState.ERROR,
    },
    {
        "trigger": "recover",
        "source": ExecutionState.ERROR,
        "dest": ExecutionState.RECOVERING,
    },
    {
        "trigger": "recovery_complete",
        "source": ExecutionState.RECOVERING,
        "dest": ExecutionState.READY,
    },
    {
        "trigger": "recovery_failed",
        "source": ExecutionState.RECOVERING,
        "dest": ExecutionState.ERROR,
    },
    {"trigger": "reset", "source": "*", "dest": ExecutionState.IDLE},
]


# ==================== ENHANCED STATE MACHINE ====================


class ExecutionStateMachine:
    """Enhanced finite state machine for execution lifecycle.

    Features:
    - State persistence during failures
    - Resource cleanup on abort
    - Complex transition validation
    - Recovery from crashed states

    Parameters
    ----------
    execution_id : str | None
        Identifier to bind into logs for traceability.
    enable_persistence : bool
        Whether to enable state persistence.
    storage_dir : Path | None
        Directory for state persistence.
    """

    def __init__(
        self,
        execution_id: str | None = None,
        enable_persistence: bool = True,
        storage_dir: Path | None = None,
    ):
        """Initialize the state machine.

        Args:
            execution_id: Optional execution identifier
            enable_persistence: Whether to persist state
            storage_dir: Optional storage directory
        """
        self.execution_id = execution_id or str(int(time.time() * 1000))
        self.logger = get_logger("state").bind(execution_id=execution_id)
        self.history: list[str] = []
        self.state: str = ExecutionState.IDLE.value
        self.context_data: dict[str, Any] = {}

        # Initialize components
        self.resource_manager = ResourceCleanupManager()
        self.transition_validator = TransitionValidator()
        self._setup_default_validation_rules()

        # Persistence
        self.enable_persistence = enable_persistence
        self._persistence: StatePersistence | None = None
        if enable_persistence:
            self._persistence = StatePersistence(storage_dir=storage_dir)

        # Error tracking
        self.last_error: Exception | None = None
        self.error_traceback: str | None = None

        # Setup state machine
        self._machine = Machine(
            model=self,
            states=[state.value for state in ExecutionState],
            transitions=TRANSITIONS,
            initial=ExecutionState.IDLE.value,
            auto_transitions=False,
            ignore_invalid_triggers=True,
            after_state_change=self._record_transition,
            send_event=False,
        )

    def _setup_default_validation_rules(self) -> None:
        """Setup default transition validation rules."""
        # Rule: Cannot execute without a plan
        self.transition_validator.add_rule(
            TransitionRule(
                trigger="execute",
                source=ExecutionState.READY.value,
                dest=ExecutionState.RUNNING.value,
                validators=[
                    lambda ctx: (
                        bool(ctx.get("has_plan", False)),
                        "Cannot execute without a plan",
                    )
                ],
                description="Require plan before execution",
            )
        )

        # Rule: Abort triggers cleanup
        self.transition_validator.add_rule(
            TransitionRule(
                trigger="abort",
                source=[ExecutionState.RUNNING.value, ExecutionState.PAUSED.value],
                dest=ExecutionState.ABORTED.value,
                after_callbacks=[lambda ctx: self._handle_abort(ctx)],
                description="Cleanup on abort",
            )
        )

    def _handle_abort(self, context: dict[str, Any]) -> None:
        """Handle abort transition - cleanup resources."""
        self.logger.info("state.abort_cleanup_started")
        stats = self.resource_manager.cleanup_all()
        self.logger.info("state.abort_cleanup_completed", **stats)

    def _record_transition(self) -> None:
        """Record state transition and persist."""
        self.history.append(self.state)
        self.logger.info("state.transition", state=self.state)

        # Persist state after transition
        if self.enable_persistence and self._persistence:
            try:
                self._persistence.persist(self._get_persisted_state())
            except StatePersistenceError as exc:
                self.logger.warning("state.persist_failed", error=str(exc))

    def _get_persisted_state(self) -> PersistedState:
        """Get current state for persistence."""
        error_info = None
        if self.last_error:
            error_info = {
                "type": type(self.last_error).__name__,
                "message": str(self.last_error),
                "traceback": self.error_traceback,
            }

        return PersistedState(
            execution_id=self.execution_id,
            state=self.state,
            history=list(self.history),
            context_data=self.context_data,
            resources=[
                r["id"] for r in self.resource_manager.get_active_resources()
            ],
            timestamp=time.time(),
            error_info=error_info,
        )

    # ==================== PUBLIC API ====================

    def set_context(self, key: str, value: Any) -> None:
        """Set context data for validation.

        Args:
            key: Context key
            value: Context value
        """
        self.context_data[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context data.

        Args:
            key: Context key
            default: Default value

        Returns:
            Context value or default
        """
        return self.context_data.get(key, default)

    def register_resource(
        self,
        resource: ResourceHandle,
        group: str | None = None,
    ) -> str:
        """Register a resource for cleanup on abort.

        Args:
            resource: Resource to register
            group: Optional group name

        Returns:
            Resource ID
        """
        return self.resource_manager.register(resource, group)

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add a cleanup callback for abort.

        Args:
            callback: Cleanup function
        """
        self.resource_manager.add_cleanup_callback(callback)

    def record_error(self, error: Exception) -> None:
        """Record an error that occurred during execution.

        Args:
            error: Exception that occurred
        """
        self.last_error = error
        self.error_traceback = traceback.format_exc()
        self.set_context("last_error", str(error))

    def validate_transition(self, trigger: str) -> tuple[bool, str]:
        """Check if a transition is valid.

        Args:
            trigger: Transition trigger

        Returns:
            Tuple of (is_valid, reason)
        """
        is_valid, reason, _ = self.transition_validator.validate_transition(
            trigger, self.state, self.context_data
        )
        return is_valid, reason

    def safe_transition(self, trigger: str) -> tuple[bool, str]:
        """Attempt a transition with validation.

        Args:
            trigger: Transition trigger

        Returns:
            Tuple of (success, reason)
        """
        is_valid, reason = self.validate_transition(trigger)
        if not is_valid:
            return False, reason

        # Attempt the transition
        trigger_method = getattr(self, trigger, None)
        if trigger_method and callable(trigger_method):
            try:
                trigger_method()
                return True, f"Transition '{trigger}' successful"
            except Exception as exc:
                return False, f"Transition failed: {exc}"

        return False, f"Unknown trigger: {trigger}"

    def recover_from_crash(self) -> bool:
        """Attempt to recover from a crashed state.

        Returns:
            True if recovery successful
        """
        if not self._persistence:
            return False

        persisted = self._persistence.load(self.execution_id)
        if not persisted:
            return False

        # Restore state
        self.state = persisted.state
        self.history = persisted.history
        self.context_data = persisted.context_data

        # Trigger recovery
        if self.state in (
            ExecutionState.RUNNING.value,
            ExecutionState.PAUSED.value,
        ):
            self.recover()
            return True

        return False

    def cleanup_on_complete(self) -> None:
        """Cleanup persistence after successful completion."""
        if self._persistence:
            self._persistence.delete(self.execution_id)

    def start_auto_persist(self) -> None:
        """Start automatic state persistence."""
        if self._persistence:
            self._persistence.start_auto_persist(self._get_persisted_state)

    def stop_auto_persist(self) -> None:
        """Stop automatic state persistence."""
        if self._persistence:
            self._persistence.stop_auto_persist()

    # ==================== CONVENIENCE ACCESSORS ====================

    @property
    def state_enum(self) -> ExecutionState:
        """Get current state as enum."""
        return ExecutionState(self.state)

    @property
    def is_running(self) -> bool:
        """Check if execution is running."""
        return self.state == ExecutionState.RUNNING.value

    @property
    def is_terminal(self) -> bool:
        """Check if in a terminal state."""
        return self.state in (
            ExecutionState.COMPLETED.value,
            ExecutionState.ABORTED.value,
            ExecutionState.ERROR.value,
        )

    @property
    def can_resume(self) -> bool:
        """Check if execution can be resumed."""
        return self.state == ExecutionState.PAUSED.value

    def snapshot(self) -> dict[str, Any]:
        """Get a snapshot of the current state."""
        return {
            "execution_id": self.execution_id,
            "state": self.state,
            "history": list(self.history),
            "context_data": self.context_data,
            "resources": self.resource_manager.get_active_resources(),
            "is_terminal": self.is_terminal,
            "last_error": str(self.last_error) if self.last_error else None,
        }


# ==================== MODULE EXPORTS ====================

__all__ = [
    # Enums
    "ExecutionState",
    # Exceptions
    "StateTransitionError",
    "StatePersistenceError",
    # Persistence
    "PersistedState",
    "StatePersistence",
    # Resources
    "ResourceHandle",
    "FileResource",
    "MemoryResource",
    "ResourceCleanupManager",
    # Validation
    "TransitionRule",
    "TransitionValidator",
    # Transitions
    "TRANSITIONS",
    # Main Class
    "ExecutionStateMachine",
]
