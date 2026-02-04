"""Execution Context for Dynamic Tool System.

This module provides the execution context that carries state information
across tool executions. The context enables tools to access:
- Current working directory
- Environment variables
- Active git repository state
- Open terminal sessions
- File system state
- Conversation history

The context also supports:
- Snapshots for rollback capability
- Serialization for persistence
- Inheritance for nested operations
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from copy import deepcopy
import hashlib


@dataclass
class GitState:
    """Current state of a git repository."""
    repo_path: Optional[str] = None
    current_branch: Optional[str] = None
    is_dirty: bool = False
    has_remote: bool = False
    remote_url: Optional[str] = None
    uncommitted_changes: List[str] = field(default_factory=list)
    recent_commits: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitState":
        return cls(**data)


@dataclass
class TerminalSession:
    """State of a terminal session."""
    session_id: str
    shell_type: str  # powershell, bash, cmd, zsh
    working_directory: str
    is_active: bool = True
    pid: Optional[int] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    environment: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FileSystemState:
    """Current state of the file system context."""
    current_directory: str
    recent_files: List[str] = field(default_factory=list)  # Recently accessed files
    recent_directories: List[str] = field(default_factory=list)  # Recently accessed directories
    watched_paths: Set[str] = field(default_factory=set)  # Paths being monitored
    temp_files: List[str] = field(default_factory=list)  # Temporary files created
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_directory": self.current_directory,
            "recent_files": self.recent_files,
            "recent_directories": self.recent_directories,
            "watched_paths": list(self.watched_paths),
            "temp_files": self.temp_files,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileSystemState":
        data = dict(data)
        data["watched_paths"] = set(data.get("watched_paths", []))
        return cls(**data)


@dataclass
class ConversationMessage:
    """A message in the conversation history."""
    role: str  # user, assistant, system, tool
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserPreferences:
    """User preferences learned from behavior."""
    preferred_shell: Optional[str] = None
    preferred_editor: Optional[str] = None
    verbosity_level: str = "normal"  # minimal, normal, verbose
    auto_confirm_low_risk: bool = False
    working_directories: List[str] = field(default_factory=list)
    recent_backends: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SystemInfo:
    """Information about the system environment."""
    os_name: str = field(default_factory=lambda: platform.system())
    os_version: str = field(default_factory=lambda: platform.version())
    python_version: str = field(default_factory=lambda: platform.python_version())
    architecture: str = field(default_factory=lambda: platform.machine())
    username: str = field(default_factory=lambda: os.getenv("USER", os.getenv("USERNAME", "unknown")))
    home_directory: str = field(default_factory=lambda: str(Path.home()))
    has_gpu: bool = False
    available_memory_gb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionContext:
    """Complete execution context for tool operations.
    
    This context is passed to every tool execution and provides
    all necessary information about the current system state.
    """
    # Unique context identifier
    context_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    
    # System state
    system_info: SystemInfo = field(default_factory=SystemInfo)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # Working context
    current_directory: str = field(default_factory=os.getcwd)
    file_system_state: FileSystemState = field(default_factory=lambda: FileSystemState(
        current_directory=os.getcwd()
    ))
    
    # Git context
    git_state: Optional[GitState] = None
    
    # Terminal context
    terminal_sessions: Dict[str, TerminalSession] = field(default_factory=dict)
    active_terminal_id: Optional[str] = None
    
    # Conversation context
    conversation_history: List[ConversationMessage] = field(default_factory=list)
    
    # User context
    user_preferences: UserPreferences = field(default_factory=UserPreferences)
    
    # Execution tracking
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    pending_operations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    parent_context_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize environment variables from system."""
        if not self.environment_variables:
            # Capture key environment variables
            important_vars = [
                "PATH", "HOME", "USER", "USERNAME", "SHELL",
                "PYTHON", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX",
                "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL",
                "GITHUB_TOKEN", "GH_TOKEN",
            ]
            for var in important_vars:
                value = os.environ.get(var)
                if value:
                    self.environment_variables[var] = value
    
    def update(self):
        """Update the timestamp."""
        self.updated_at = datetime.now().isoformat()
    
    def set_current_directory(self, path: str):
        """Change the current directory."""
        path = str(Path(path).resolve())
        self.current_directory = path
        self.file_system_state.current_directory = path
        
        # Track in recent directories
        if path not in self.file_system_state.recent_directories:
            self.file_system_state.recent_directories.insert(0, path)
            # Keep only last 20
            self.file_system_state.recent_directories = self.file_system_state.recent_directories[:20]
        
        # Actually change the working directory
        try:
            os.chdir(path)
        except Exception:
            pass
        
        self.update()
    
    def add_recent_file(self, path: str):
        """Track a recently accessed file."""
        path = str(Path(path).resolve())
        if path not in self.file_system_state.recent_files:
            self.file_system_state.recent_files.insert(0, path)
            # Keep only last 50
            self.file_system_state.recent_files = self.file_system_state.recent_files[:50]
        self.update()
    
    def add_conversation_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to conversation history."""
        self.conversation_history.append(ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
        # Keep reasonable history size
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]
        self.update()
    
    def get_recent_conversation(self, count: int = 10) -> List[ConversationMessage]:
        """Get recent conversation messages."""
        return self.conversation_history[-count:]
    
    def add_execution_record(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """Record a tool execution."""
        self.execution_history.append({
            "tool_name": tool_name,
            "parameters": parameters,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })
        # Keep reasonable history
        if len(self.execution_history) > 200:
            self.execution_history = self.execution_history[-200:]
        self.update()
    
    def update_git_state(self, state: GitState):
        """Update the git state."""
        self.git_state = state
        self.update()
    
    def add_terminal_session(self, session: TerminalSession):
        """Add a terminal session."""
        self.terminal_sessions[session.session_id] = session
        self.active_terminal_id = session.session_id
        self.update()
    
    def get_active_terminal(self) -> Optional[TerminalSession]:
        """Get the active terminal session."""
        if self.active_terminal_id:
            return self.terminal_sessions.get(self.active_terminal_id)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            "context_id": self.context_id,
            "system_info": self.system_info.to_dict(),
            "environment_variables": self.environment_variables,
            "current_directory": self.current_directory,
            "file_system_state": self.file_system_state.to_dict(),
            "git_state": self.git_state.to_dict() if self.git_state else None,
            "terminal_sessions": {
                k: v.to_dict() for k, v in self.terminal_sessions.items()
            },
            "active_terminal_id": self.active_terminal_id,
            "conversation_history": [m.to_dict() for m in self.conversation_history],
            "user_preferences": self.user_preferences.to_dict(),
            "execution_history": self.execution_history,
            "pending_operations": self.pending_operations,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parent_context_id": self.parent_context_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionContext":
        """Deserialize context from dictionary."""
        context = cls(
            context_id=data.get("context_id", ""),
            environment_variables=data.get("environment_variables", {}),
            current_directory=data.get("current_directory", os.getcwd()),
            active_terminal_id=data.get("active_terminal_id"),
            execution_history=data.get("execution_history", []),
            pending_operations=data.get("pending_operations", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            parent_context_id=data.get("parent_context_id"),
        )
        
        if data.get("system_info"):
            context.system_info = SystemInfo(**data["system_info"])
        
        if data.get("file_system_state"):
            context.file_system_state = FileSystemState.from_dict(data["file_system_state"])
        
        if data.get("git_state"):
            context.git_state = GitState.from_dict(data["git_state"])
        
        if data.get("terminal_sessions"):
            context.terminal_sessions = {
                k: TerminalSession(**v) for k, v in data["terminal_sessions"].items()
            }
        
        if data.get("conversation_history"):
            context.conversation_history = [
                ConversationMessage(**m) for m in data["conversation_history"]
            ]
        
        if data.get("user_preferences"):
            context.user_preferences = UserPreferences(**data["user_preferences"])
        
        return context
    
    def create_child_context(self) -> "ExecutionContext":
        """Create a child context for nested operations."""
        child = ExecutionContext(
            system_info=self.system_info,
            environment_variables=dict(self.environment_variables),
            current_directory=self.current_directory,
            file_system_state=FileSystemState(
                current_directory=self.current_directory,
                recent_files=list(self.file_system_state.recent_files),
                recent_directories=list(self.file_system_state.recent_directories),
            ),
            git_state=deepcopy(self.git_state) if self.git_state else None,
            terminal_sessions=dict(self.terminal_sessions),
            active_terminal_id=self.active_terminal_id,
            conversation_history=list(self.conversation_history),
            user_preferences=deepcopy(self.user_preferences),
            parent_context_id=self.context_id,
        )
        return child
    
    def to_llm_context(self) -> str:
        """Generate a context string for LLM understanding.
        
        This creates a natural language description of the current state
        that helps the LLM make informed decisions.
        """
        ctx = []
        ctx.append(f"Current Directory: {self.current_directory}")
        ctx.append(f"Operating System: {self.system_info.os_name} {self.system_info.os_version}")
        
        if self.git_state and self.git_state.repo_path:
            ctx.append(f"Git Repository: {self.git_state.repo_path}")
            ctx.append(f"Current Branch: {self.git_state.current_branch or 'Unknown'}")
            if self.git_state.is_dirty:
                ctx.append("Repository has uncommitted changes")
        
        if self.file_system_state.recent_files:
            ctx.append(f"Recently accessed files: {', '.join(self.file_system_state.recent_files[:5])}")
        
        if self.terminal_sessions:
            ctx.append(f"Active terminal sessions: {len(self.terminal_sessions)}")
        
        return "\n".join(ctx)


@dataclass
class ContextSnapshot:
    """A snapshot of the execution context for rollback."""
    snapshot_id: str
    context_data: Dict[str, Any]
    created_at: str
    description: str = ""
    
    @classmethod
    def create(cls, context: ExecutionContext, description: str = "") -> "ContextSnapshot":
        """Create a snapshot from a context."""
        data = context.to_dict()
        snapshot_id = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        
        return cls(
            snapshot_id=snapshot_id,
            context_data=data,
            created_at=datetime.now().isoformat(),
            description=description,
        )
    
    def restore(self) -> ExecutionContext:
        """Restore the context from this snapshot."""
        return ExecutionContext.from_dict(self.context_data)


class ContextManager:
    """Manager for execution contexts with snapshot support.
    
    Provides:
    - Context creation and management
    - Snapshot creation and restoration
    - Context persistence
    """
    
    def __init__(self, persistence_path: Optional[Path] = None):
        """Initialize the context manager.
        
        Args:
            persistence_path: Optional path for context persistence
        """
        self._current_context: Optional[ExecutionContext] = None
        self._snapshots: Dict[str, ContextSnapshot] = {}
        self._persistence_path = persistence_path
        
        # Load persisted context if available
        if persistence_path and persistence_path.exists():
            self._load_persisted_context()
    
    @property
    def current_context(self) -> ExecutionContext:
        """Get or create the current context."""
        if self._current_context is None:
            self._current_context = ExecutionContext()
        return self._current_context
    
    def create_context(self) -> ExecutionContext:
        """Create a new execution context."""
        self._current_context = ExecutionContext()
        return self._current_context
    
    def create_snapshot(self, description: str = "") -> ContextSnapshot:
        """Create a snapshot of the current context."""
        snapshot = ContextSnapshot.create(self.current_context, description)
        self._snapshots[snapshot.snapshot_id] = snapshot
        return snapshot
    
    def restore_snapshot(self, snapshot_id: str) -> Optional[ExecutionContext]:
        """Restore context from a snapshot."""
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot:
            self._current_context = snapshot.restore()
            return self._current_context
        return None
    
    def get_snapshots(self) -> List[ContextSnapshot]:
        """Get all available snapshots."""
        return list(self._snapshots.values())
    
    def persist(self):
        """Persist the current context to disk."""
        if self._persistence_path and self._current_context:
            data = {
                "context": self._current_context.to_dict(),
                "snapshots": {
                    k: {
                        "snapshot_id": v.snapshot_id,
                        "context_data": v.context_data,
                        "created_at": v.created_at,
                        "description": v.description,
                    }
                    for k, v in self._snapshots.items()
                }
            }
            self._persistence_path.write_text(json.dumps(data, indent=2, default=str))
    
    def _load_persisted_context(self):
        """Load persisted context from disk."""
        try:
            data = json.loads(self._persistence_path.read_text())
            if data.get("context"):
                self._current_context = ExecutionContext.from_dict(data["context"])
            if data.get("snapshots"):
                for k, v in data["snapshots"].items():
                    self._snapshots[k] = ContextSnapshot(
                        snapshot_id=v["snapshot_id"],
                        context_data=v["context_data"],
                        created_at=v["created_at"],
                        description=v.get("description", ""),
                    )
        except Exception:
            pass  # Start fresh if persistence fails


# Global context manager instance
_global_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get the global context manager instance."""
    global _global_context_manager
    if _global_context_manager is None:
        _global_context_manager = ContextManager()
    return _global_context_manager


def get_current_context() -> ExecutionContext:
    """Get the current execution context."""
    return get_context_manager().current_context
