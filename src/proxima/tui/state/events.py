"""Custom event definitions for Proxima TUI.

Dataclass-based events for the TUI event bus system.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime

from textual.message import Message


# ============================================================
# Base Event Class
# ============================================================

class TUIEvent(Message):
    """Base class for all TUI events."""
    pass


# ============================================================
# Execution Events
# ============================================================

@dataclass
class ExecutionStarted(TUIEvent):
    """Fired when execution begins."""
    
    task: str
    backend: str
    simulator: str
    session_id: str
    qubits: int = 0
    shots: int = 0


@dataclass
class ExecutionProgress(TUIEvent):
    """Fired on progress update."""
    
    progress: float
    stage: str
    stage_index: int
    total_stages: int
    elapsed_ms: float
    eta_ms: Optional[float] = None


@dataclass
class StageStarted(TUIEvent):
    """Fired when a stage starts."""
    
    stage_name: str
    stage_index: int


@dataclass
class StageCompleted(TUIEvent):
    """Fired when a stage completes."""
    
    stage_name: str
    stage_index: int
    duration_ms: float
    success: bool


@dataclass
class ExecutionCompleted(TUIEvent):
    """Fired when execution finishes successfully."""
    
    result: Dict[str, Any]
    total_time_ms: float


@dataclass
class ExecutionFailed(TUIEvent):
    """Fired when execution fails."""
    
    error: str
    stage: str
    partial_result: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionPaused(TUIEvent):
    """Fired when execution is paused."""
    
    checkpoint_id: str
    stage_index: int


@dataclass
class ExecutionResumed(TUIEvent):
    """Fired when execution resumes."""
    
    checkpoint_id: str
    stage_index: int


@dataclass
class ExecutionAborted(TUIEvent):
    """Fired when execution is aborted."""
    
    reason: str


# ============================================================
# Memory Events
# ============================================================

@dataclass
class MemoryUpdate(TUIEvent):
    """Fired on memory sample."""
    
    percent: float
    level: str
    used_mb: float
    available_mb: float


@dataclass
class MemoryAlert(TUIEvent):
    """Fired when memory threshold crossed."""
    
    previous_level: str
    current_level: str
    percent: float
    message: str


# ============================================================
# Backend Events
# ============================================================

@dataclass
class BackendHealthChanged(TUIEvent):
    """Fired when backend health changes."""
    
    backend: str
    previous_status: str
    current_status: str
    response_time_ms: Optional[float] = None


@dataclass
class BackendSelected(TUIEvent):
    """Fired when a backend is selected."""
    
    backend: str
    simulator: str


# ============================================================
# Session Events
# ============================================================

@dataclass
class SessionCreated(TUIEvent):
    """Fired when new session created."""
    
    session_id: str
    title: str


@dataclass
class SessionSwitched(TUIEvent):
    """Fired when session is switched."""
    
    previous_id: Optional[str]
    new_id: str
    title: str


@dataclass
class SessionEnded(TUIEvent):
    """Fired when session ends."""
    
    session_id: str
    status: str  # completed, aborted, error


# ============================================================
# Checkpoint Events
# ============================================================

@dataclass
class CheckpointCreated(TUIEvent):
    """Fired when checkpoint is created."""
    
    checkpoint_id: str
    stage_index: int
    timestamp: float


@dataclass
class RollbackStarted(TUIEvent):
    """Fired when rollback begins."""
    
    checkpoint_id: str
    target_stage: int


@dataclass
class RollbackCompleted(TUIEvent):
    """Fired when rollback completes."""
    
    checkpoint_id: str
    stage_index: int


# ============================================================
# Consent Events
# ============================================================

@dataclass
class ConsentRequested(TUIEvent):
    """Fired when consent is needed."""
    
    consent_type: str  # memory, backend, llm_data, abort, rollback
    details: Dict[str, Any]
    callback_id: str


@dataclass
class ConsentResponse(TUIEvent):
    """Fired when user responds to consent."""
    
    callback_id: str
    action: str  # allow, allow_session, deny


# ============================================================
# LLM Events
# ============================================================

@dataclass
class LLMConnected(TUIEvent):
    """Fired when LLM is connected."""
    
    provider: str
    model: str


@dataclass
class LLMDisconnected(TUIEvent):
    """Fired when LLM is disconnected."""
    
    reason: str


@dataclass
class LLMModelChanged(TUIEvent):
    """Fired when LLM model is changed."""
    
    previous_model: Optional[str]
    new_model: str
    provider: str


@dataclass
class LLMTokenUpdate(TUIEvent):
    """Fired when token count updates."""
    
    prompt_tokens: int
    completion_tokens: int
    total_cost: float


# ============================================================
# Navigation Events
# ============================================================

@dataclass
class ScreenChanged(TUIEvent):
    """Fired when screen changes."""
    
    previous_screen: str
    new_screen: str


@dataclass
class DialogOpened(TUIEvent):
    """Fired when a dialog opens."""
    
    dialog_type: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class DialogClosed(TUIEvent):
    """Fired when a dialog closes."""
    
    dialog_type: str
    result: Optional[Any] = None


# ============================================================
# Log Events
# ============================================================

@dataclass
class LogEntry(TUIEvent):
    """Fired when a log entry is added."""
    
    level: str  # debug, info, warning, error
    message: str
    timestamp: Optional[datetime] = None
    source: Optional[str] = None


# ============================================================
# Command Events
# ============================================================

@dataclass
class CommandExecuted(TUIEvent):
    """Fired when a command is executed."""
    
    command_id: str
    command_name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
