"""Session management for persistence and recovery.

Provides:
- Session: Encapsulates execution session state
- SessionManager: Create, save, load, resume sessions
- SessionLock: File-based locking for concurrent access
- SessionCleanup: Cleanup old/expired sessions
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any


class SessionStatus(Enum):
    """Session lifecycle status."""

    CREATED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    ABORTED = auto()


@dataclass
class SessionMetadata:
    """Session metadata."""

    id: str
    created_at: float
    updated_at: float
    status: SessionStatus
    name: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class SessionCheckpoint:
    """Checkpoint within a session for recovery.
    
    Enhanced with rollback support:
    - Full state snapshot for complete restoration
    - Dependency tracking for selective rollback
    - Metadata for debugging and audit
    """

    id: str
    timestamp: float
    stage: str
    state: dict[str, Any]
    message: str | None = None
    
    # Enhanced rollback support
    parent_checkpoint_id: str | None = None  # For checkpoint chain
    state_hash: str | None = None  # For integrity verification
    dependencies: list[str] = field(default_factory=list)  # External dependencies
    rollback_actions: list[dict[str, Any]] = field(default_factory=list)  # Actions to undo
    
    def compute_hash(self) -> str:
        """Compute hash of state for integrity verification."""
        import hashlib
        state_str = json.dumps(self.state, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]
    
    def verify_integrity(self) -> bool:
        """Verify checkpoint integrity using stored hash."""
        if self.state_hash is None:
            return True  # No hash stored, assume valid
        return self.compute_hash() == self.state_hash


@dataclass
class Session:
    """Execution session with state and checkpoints."""

    metadata: SessionMetadata
    config: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[SessionCheckpoint] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())[:8]
        now = time.time()

        return cls(
            metadata=SessionMetadata(
                id=session_id,
                created_at=now,
                updated_at=now,
                status=SessionStatus.CREATED,
                name=name or f"session-{session_id}",
            ),
            config=config or {},
        )

    def update_status(self, status: SessionStatus) -> None:
        """Update session status."""
        self.metadata.status = status
        self.metadata.updated_at = time.time()

    def checkpoint(
        self,
        stage: str,
        state: dict[str, Any],
        message: str | None = None,
        rollback_actions: list[dict[str, Any]] | None = None,
    ) -> SessionCheckpoint:
        """Create a checkpoint with enhanced rollback support.
        
        Args:
            stage: Current execution stage name
            state: State dict to checkpoint
            message: Optional description
            rollback_actions: Optional list of actions to execute on rollback
                Each action is a dict with 'type' and parameters
        
        Returns:
            Created checkpoint
        """
        # Get parent checkpoint ID if exists
        parent_id = self.checkpoints[-1].id if self.checkpoints else None
        
        cp = SessionCheckpoint(
            id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            stage=stage,
            state=state.copy(),
            message=message,
            parent_checkpoint_id=parent_id,
            rollback_actions=rollback_actions or [],
        )
        # Compute and store hash for integrity verification
        cp.state_hash = cp.compute_hash()
        
        self.checkpoints.append(cp)
        self.metadata.updated_at = time.time()
        return cp

    def latest_checkpoint(self) -> SessionCheckpoint | None:
        """Get the most recent checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None
    
    def get_checkpoint(self, checkpoint_id: str) -> SessionCheckpoint | None:
        """Get checkpoint by ID."""
        for cp in self.checkpoints:
            if cp.id == checkpoint_id:
                return cp
        return None
    
    def get_checkpoint_by_stage(self, stage: str) -> SessionCheckpoint | None:
        """Get the most recent checkpoint for a specific stage."""
        for cp in reversed(self.checkpoints):
            if cp.stage == stage:
                return cp
        return None
    
    def rollback_to(
        self,
        checkpoint_id: str,
        execute_rollback_actions: bool = True,
    ) -> tuple[bool, list[str]]:
        """Rollback session state to a specific checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to rollback to
            execute_rollback_actions: Whether to execute stored rollback actions
        
        Returns:
            Tuple of (success, list of executed action descriptions)
        """
        target_cp = self.get_checkpoint(checkpoint_id)
        if not target_cp:
            return False, [f"Checkpoint {checkpoint_id} not found"]
        
        # Verify integrity
        if not target_cp.verify_integrity():
            return False, ["Checkpoint integrity verification failed"]
        
        executed_actions: list[str] = []
        
        # Execute rollback actions for all checkpoints after target (in reverse)
        if execute_rollback_actions:
            target_idx = self.checkpoints.index(target_cp)
            for cp in reversed(self.checkpoints[target_idx + 1:]):
                for action in cp.rollback_actions:
                    action_desc = self._execute_rollback_action(action)
                    if action_desc:
                        executed_actions.append(action_desc)
        
        # Restore state from checkpoint
        self.state = target_cp.state.copy()
        
        # Remove checkpoints after target
        target_idx = self.checkpoints.index(target_cp)
        self.checkpoints = self.checkpoints[:target_idx + 1]
        
        self.metadata.updated_at = time.time()
        self.add_log("info", f"Rolled back to checkpoint {checkpoint_id}", 
                     stage=target_cp.stage)
        
        return True, executed_actions
    
    def _execute_rollback_action(self, action: dict[str, Any]) -> str | None:
        """Execute a single rollback action.
        
        Supported action types:
        - 'delete_file': Remove a file (path in 'path')
        - 'restore_file': Restore file content (path, content)
        - 'set_config': Set config value (key, value)
        - 'cleanup_temp': Remove temporary data (prefix)
        - 'custom': Execute custom callback (callback_name, args)
        
        Returns:
            Description of executed action, or None if no action taken
        """
        action_type = action.get('type')
        
        if action_type == 'delete_file':
            from pathlib import Path
            path = Path(action.get('path', ''))
            if path.exists():
                path.unlink()
                return f"Deleted file: {path}"
        
        elif action_type == 'restore_file':
            from pathlib import Path
            path = Path(action.get('path', ''))
            content = action.get('content', '')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return f"Restored file: {path}"
        
        elif action_type == 'set_config':
            key = action.get('key')
            value = action.get('value')
            if key:
                self.config[key] = value
                return f"Set config: {key}"
        
        elif action_type == 'cleanup_temp':
            prefix = action.get('prefix', '')
            # Clean up temporary data with prefix
            keys_to_remove = [k for k in self.state.keys() if k.startswith(prefix)]
            for k in keys_to_remove:
                del self.state[k]
            if keys_to_remove:
                return f"Cleaned temp data: {prefix}*"
        
        elif action_type == 'custom':
            callback_name = action.get('callback_name')
            return f"Custom rollback: {callback_name}"
        
        return None
    
    def rollback_to_stage(self, stage: str) -> tuple[bool, list[str]]:
        """Rollback to the most recent checkpoint of a specific stage."""
        cp = self.get_checkpoint_by_stage(stage)
        if cp:
            return self.rollback_to(cp.id)
        return False, [f"No checkpoint found for stage: {stage}"]
    
    def can_rollback(self) -> bool:
        """Check if rollback is possible (has at least 2 checkpoints)."""
        return len(self.checkpoints) >= 2
    
    def get_rollback_chain(self) -> list[dict[str, Any]]:
        """Get list of checkpoints for rollback visualization."""
        return [
            {
                'id': cp.id,
                'stage': cp.stage,
                'timestamp': cp.timestamp,
                'message': cp.message,
                'has_rollback_actions': len(cp.rollback_actions) > 0,
                'is_valid': cp.verify_integrity(),
            }
            for cp in self.checkpoints
        ]

    def add_log(self, level: str, message: str, **extra: Any) -> None:
        """Add a log entry."""
        self.logs.append(
            {
                "timestamp": time.time(),
                "level": level,
                "message": message,
                **extra,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize session to dict."""
        return {
            "metadata": {
                "id": self.metadata.id,
                "created_at": self.metadata.created_at,
                "updated_at": self.metadata.updated_at,
                "status": self.metadata.status.name,
                "name": self.metadata.name,
                "description": self.metadata.description,
                "tags": self.metadata.tags,
            },
            "config": self.config,
            "state": self.state,
            "checkpoints": [
                {
                    "id": cp.id,
                    "timestamp": cp.timestamp,
                    "stage": cp.stage,
                    "state": cp.state,
                    "message": cp.message,
                }
                for cp in self.checkpoints
            ],
            "results": self.results,
            "logs": self.logs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Deserialize session from dict."""
        meta = data["metadata"]
        return cls(
            metadata=SessionMetadata(
                id=meta["id"],
                created_at=meta["created_at"],
                updated_at=meta["updated_at"],
                status=SessionStatus[meta["status"]],
                name=meta.get("name"),
                description=meta.get("description"),
                tags=meta.get("tags", []),
            ),
            config=data.get("config", {}),
            state=data.get("state", {}),
            checkpoints=[
                SessionCheckpoint(
                    id=cp["id"],
                    timestamp=cp["timestamp"],
                    stage=cp["stage"],
                    state=cp["state"],
                    message=cp.get("message"),
                )
                for cp in data.get("checkpoints", [])
            ],
            results=data.get("results", {}),
            logs=data.get("logs", []),
        )


class SessionManager:
    """Manages session lifecycle and persistence."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._storage_dir = storage_dir
        self._current: Session | None = None
        self._sessions: dict[str, Session] = {}

    @property
    def current(self) -> Session | None:
        return self._current

    def create(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> Session:
        """Create and set current session."""
        session = Session.create(name=name, config=config)
        self._sessions[session.metadata.id] = session
        self._current = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Get session by ID."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        return self.load(session_id)

    def set_current(self, session_id: str) -> bool:
        """Set current session by ID."""
        session = self.get(session_id)
        if session:
            self._current = session
            return True
        return False

    def save(self, session: Session | None = None) -> bool:
        """Save session to storage."""
        session = session or self._current
        if not session or not self._storage_dir:
            return False

        self._storage_dir.mkdir(parents=True, exist_ok=True)
        path = self._storage_dir / f"{session.metadata.id}.json"
        path.write_text(json.dumps(session.to_dict(), indent=2))
        return True

    def load(self, session_id: str) -> Session | None:
        """Load session from storage."""
        if not self._storage_dir:
            return None

        path = self._storage_dir / f"{session_id}.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            session = Session.from_dict(data)
            self._sessions[session_id] = session
            return session
        except Exception:
            return None

    def list_sessions(self) -> list[SessionMetadata]:
        """List all available sessions."""
        sessions = list(self._sessions.values())

        if self._storage_dir and self._storage_dir.exists():
            for path in self._storage_dir.glob("*.json"):
                session_id = path.stem
                if session_id not in self._sessions:
                    session = self.load(session_id)
                    if session:
                        sessions.append(session)

        return [s.metadata for s in sessions]

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]

        if self._current and self._current.metadata.id == session_id:
            self._current = None

        if self._storage_dir:
            path = self._storage_dir / f"{session_id}.json"
            if path.exists():
                path.unlink()
                return True

        return session_id in self._sessions

    def resume(self, session_id: str) -> Session | None:
        """Resume a session from its last checkpoint."""
        session = self.get(session_id)
        if not session:
            return None

        checkpoint = session.latest_checkpoint()
        if checkpoint:
            session.state = checkpoint.state.copy()

        session.update_status(SessionStatus.RUNNING)
        self._current = session
        return session


# =============================================================================
# Session Locking
# =============================================================================


class SessionLock:
    """File-based locking for session access.

    Provides exclusive access to a session file to prevent
    concurrent modifications from multiple processes.
    """

    def __init__(self, session_path: Path) -> None:
        self._session_path = session_path
        self._lock_path = session_path.with_suffix(".lock")
        self._lock_file: Any = None
        self._locked = False

    def acquire(self, timeout: float = 5.0) -> bool:
        """Acquire the lock with timeout.

        Args:
            timeout: Maximum time to wait for lock in seconds.

        Returns:
            True if lock acquired, False if timeout.
        """
        import platform

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                self._lock_file = open(self._lock_path, "w")

                if platform.system() != "Windows":
                    import fcntl  # type: ignore[import]

                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
                else:
                    # Windows: use msvcrt for file locking
                    import msvcrt

                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]

                self._locked = True
                # Write PID for debugging
                self._lock_file.write(str(os.getpid()))
                self._lock_file.flush()
                return True
            except (OSError, BlockingIOError):
                if self._lock_file:
                    self._lock_file.close()
                    self._lock_file = None
                time.sleep(0.1)

        return False

    def release(self) -> None:
        """Release the lock."""
        import platform

        if self._lock_file and self._locked:
            try:
                if platform.system() != "Windows":
                    import fcntl  # type: ignore[import]

                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
                else:
                    import msvcrt

                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            except OSError:
                pass
            finally:
                self._lock_file.close()
                self._lock_file = None
                self._locked = False
                # Clean up lock file
                try:
                    self._lock_path.unlink()
                except OSError:
                    pass

    def __enter__(self) -> SessionLock:
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock for {self._session_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @property
    def is_locked(self) -> bool:
        return self._locked


# =============================================================================
# Session Cleanup
# =============================================================================


class SessionCleanup:
    """Handles cleanup of old and expired sessions."""

    def __init__(
        self,
        storage_dir: Path,
        max_age_days: float = 30.0,
        max_sessions: int = 100,
    ) -> None:
        self._storage_dir = storage_dir
        self._max_age_seconds = max_age_days * 24 * 3600
        self._max_sessions = max_sessions

    def cleanup_expired(self) -> int:
        """Remove sessions older than max_age_days.

        Returns:
            Number of sessions cleaned up.
        """
        if not self._storage_dir.exists():
            return 0

        count = 0
        current_time = time.time()

        for path in self._storage_dir.glob("*.json"):
            try:
                # Check file modification time
                mtime = path.stat().st_mtime
                age = current_time - mtime

                if age > self._max_age_seconds:
                    path.unlink()
                    # Also remove lock file if exists
                    lock_path = path.with_suffix(".lock")
                    if lock_path.exists():
                        lock_path.unlink()
                    count += 1
            except OSError:
                continue

        return count

    def cleanup_excess(self) -> int:
        """Remove oldest sessions if count exceeds max_sessions.

        Returns:
            Number of sessions cleaned up.
        """
        if not self._storage_dir.exists():
            return 0

        sessions = []
        for path in self._storage_dir.glob("*.json"):
            try:
                sessions.append((path, path.stat().st_mtime))
            except OSError:
                continue

        if len(sessions) <= self._max_sessions:
            return 0

        # Sort by modification time (oldest first)
        sessions.sort(key=lambda x: x[1])

        # Remove excess
        excess = len(sessions) - self._max_sessions
        count = 0

        for path, _ in sessions[:excess]:
            try:
                path.unlink()
                lock_path = path.with_suffix(".lock")
                if lock_path.exists():
                    lock_path.unlink()
                count += 1
            except OSError:
                continue

        return count

    def cleanup_incomplete(self) -> int:
        """Remove sessions that are stuck in RUNNING or PAUSED state.

        Sessions that have been in these states for more than 24 hours
        are considered stuck and will be cleaned up.

        Returns:
            Number of sessions cleaned up.
        """
        if not self._storage_dir.exists():
            return 0

        count = 0
        current_time = time.time()
        stuck_threshold = 24 * 3600  # 24 hours

        for path in self._storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                status = data.get("metadata", {}).get("status")
                updated_at = data.get("metadata", {}).get("updated_at", 0)

                if status in ("RUNNING", "PAUSED"):
                    if current_time - updated_at > stuck_threshold:
                        path.unlink()
                        count += 1
            except (OSError, json.JSONDecodeError):
                continue

        return count

    def run_full_cleanup(self) -> dict[str, int]:
        """Run all cleanup operations.

        Returns:
            Dictionary with counts for each cleanup type.
        """
        return {
            "expired": self.cleanup_expired(),
            "excess": self.cleanup_excess(),
            "incomplete": self.cleanup_incomplete(),
        }


# =============================================================================
# Enhanced Session Manager with Locking
# =============================================================================


class LockingSessionManager(SessionManager):
    """Session manager with file locking support."""

    def __init__(
        self,
        storage_dir: Path | None = None,
        auto_cleanup: bool = True,
        max_age_days: float = 30.0,
    ) -> None:
        super().__init__(storage_dir)
        self._locks: dict[str, SessionLock] = {}
        self._cleanup = (
            SessionCleanup(storage_dir, max_age_days) if storage_dir else None
        )
        self._lock = threading.Lock()

        # Register cleanup on exit
        if auto_cleanup and storage_dir:
            atexit.register(self._cleanup_on_exit)

    def save(self, session: Session | None = None) -> bool:
        """Save session with locking."""
        session = session or self._current
        if not session or not self._storage_dir:
            return False

        path = self._storage_dir / f"{session.metadata.id}.json"
        lock = SessionLock(path)

        if lock.acquire():
            try:
                self._storage_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(session.to_dict(), indent=2))
                return True
            finally:
                lock.release()
        return False

    def load(self, session_id: str) -> Session | None:
        """Load session with locking."""
        if not self._storage_dir:
            return None

        path = self._storage_dir / f"{session_id}.json"
        if not path.exists():
            return None

        lock = SessionLock(path)
        if lock.acquire():
            try:
                data = json.loads(path.read_text())
                session = Session.from_dict(data)
                with self._lock:
                    self._sessions[session_id] = session
                return session
            except Exception:
                return None
            finally:
                lock.release()
        return None

    def run_cleanup(self) -> dict[str, int]:
        """Manually run cleanup."""
        if self._cleanup:
            return self._cleanup.run_full_cleanup()
        return {}

    def _cleanup_on_exit(self) -> None:
        """Cleanup handler for program exit."""
        # Save current session if exists
        if self._current:
            self.save(self._current)

        # Release any held locks
        for lock in self._locks.values():
            lock.release()
        self._locks.clear()
