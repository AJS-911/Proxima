"""Progress Tracker for Dynamic Tool System.

This module provides progress tracking capabilities for tool execution,
enabling real-time feedback to users about operation status.

Features:
- Event-based progress updates
- Status tracking for operations
- Progress percentage estimation
- Time estimation
- Progress callbacks for UI integration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class ProgressStatus(Enum):
    """Status of a tracked operation."""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"
    PAUSED = "paused"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressEventType(Enum):
    """Types of progress events."""
    STARTED = "started"
    PROGRESS_UPDATE = "progress_update"
    STATUS_CHANGE = "status_change"
    MESSAGE = "message"
    WARNING = "warning"
    ERROR = "error"
    COMPLETED = "completed"
    SUBTASK_STARTED = "subtask_started"
    SUBTASK_COMPLETED = "subtask_completed"


@dataclass
class ProgressEvent:
    """A progress event."""
    event_id: str
    event_type: ProgressEventType
    operation_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    message: str = ""
    progress_percent: Optional[float] = None
    status: Optional[ProgressStatus] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "operation_id": self.operation_id,
            "timestamp": self.timestamp,
            "message": self.message,
            "progress_percent": self.progress_percent,
            "status": self.status.value if self.status else None,
            "metadata": self.metadata,
        }


@dataclass
class TrackedOperation:
    """An operation being tracked."""
    operation_id: str
    name: str
    description: str
    status: ProgressStatus = ProgressStatus.NOT_STARTED
    progress_percent: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_duration_seconds: Optional[float] = None
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    events: List[ProgressEvent] = field(default_factory=list)
    subtasks: List["TrackedOperation"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "progress_percent": self.progress_percent,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "metadata": self.metadata,
            "error_message": self.error_message,
        }
    
    def update_progress(self, percent: float, step: str = ""):
        """Update progress percentage and current step."""
        self.progress_percent = min(100.0, max(0.0, percent))
        if step:
            self.current_step = step
        if self.status == ProgressStatus.NOT_STARTED:
            self.status = ProgressStatus.IN_PROGRESS
    
    def get_elapsed_time(self) -> Optional[float]:
        """Get elapsed time in seconds."""
        if not self.started_at:
            return None
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(self.completed_at) if self.completed_at else datetime.now()
        return (end - start).total_seconds()
    
    def get_estimated_remaining(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        if not self.started_at or self.progress_percent <= 0:
            return self.estimated_duration_seconds
        
        elapsed = self.get_elapsed_time()
        if elapsed is None or elapsed <= 0:
            return None
        
        # Estimate total time based on progress
        if self.progress_percent < 100:
            total_estimated = elapsed / (self.progress_percent / 100.0)
            return total_estimated - elapsed
        return 0.0


class ProgressTracker:
    """Tracks progress of tool executions.
    
    The tracker provides:
    - Operation tracking with status updates
    - Progress percentage tracking
    - Event emission for UI updates
    - Time estimation
    - Nested subtask tracking
    """
    
    def __init__(self):
        """Initialize the progress tracker."""
        self._operations: Dict[str, TrackedOperation] = {}
        self._event_counter = 0
        self._callbacks: List[Callable[[ProgressEvent], None]] = []
        self._operation_history: List[TrackedOperation] = []
    
    def start_operation(
        self,
        operation_id: str,
        name: str,
        description: str = "",
        total_steps: int = 0,
        estimated_duration: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrackedOperation:
        """Start tracking an operation.
        
        Args:
            operation_id: Unique identifier for the operation
            name: Operation name
            description: Operation description
            total_steps: Total number of steps (for step-based progress)
            estimated_duration: Estimated duration in seconds
            metadata: Additional metadata
            
        Returns:
            The tracked operation
        """
        operation = TrackedOperation(
            operation_id=operation_id,
            name=name,
            description=description,
            status=ProgressStatus.INITIALIZING,
            started_at=datetime.now().isoformat(),
            total_steps=total_steps,
            estimated_duration_seconds=estimated_duration,
            metadata=metadata or {},
        )
        
        self._operations[operation_id] = operation
        
        # Emit started event
        self._emit_event(ProgressEvent(
            event_id=self._next_event_id(),
            event_type=ProgressEventType.STARTED,
            operation_id=operation_id,
            message=f"Started: {name}",
            status=ProgressStatus.INITIALIZING,
            metadata={"name": name, "description": description},
        ))
        
        return operation
    
    def update_progress(
        self,
        operation_id: str,
        progress_percent: Optional[float] = None,
        current_step: Optional[str] = None,
        completed_steps: Optional[int] = None,
        message: Optional[str] = None,
    ):
        """Update operation progress.
        
        Args:
            operation_id: Operation identifier
            progress_percent: Progress percentage (0-100)
            current_step: Current step description
            completed_steps: Number of completed steps
            message: Optional progress message
        """
        operation = self._operations.get(operation_id)
        if not operation:
            logger.warning(f"Operation not found: {operation_id}")
            return
        
        if operation.status == ProgressStatus.INITIALIZING:
            operation.status = ProgressStatus.IN_PROGRESS
        
        if progress_percent is not None:
            operation.progress_percent = min(100.0, max(0.0, progress_percent))
        
        if current_step:
            operation.current_step = current_step
        
        if completed_steps is not None:
            operation.completed_steps = completed_steps
            # Update progress based on steps if percent not provided
            if progress_percent is None and operation.total_steps > 0:
                operation.progress_percent = (completed_steps / operation.total_steps) * 100
        
        # Emit progress event
        self._emit_event(ProgressEvent(
            event_id=self._next_event_id(),
            event_type=ProgressEventType.PROGRESS_UPDATE,
            operation_id=operation_id,
            message=message or current_step or "",
            progress_percent=operation.progress_percent,
            status=operation.status,
        ))
    
    def update_status(
        self,
        operation_id: str,
        status: ProgressStatus,
        message: str = "",
    ):
        """Update operation status.
        
        Args:
            operation_id: Operation identifier
            status: New status
            message: Status message
        """
        operation = self._operations.get(operation_id)
        if not operation:
            return
        
        old_status = operation.status
        operation.status = status
        
        self._emit_event(ProgressEvent(
            event_id=self._next_event_id(),
            event_type=ProgressEventType.STATUS_CHANGE,
            operation_id=operation_id,
            message=message,
            status=status,
            metadata={"old_status": old_status.value},
        ))
    
    def add_message(
        self,
        operation_id: str,
        message: str,
        level: str = "info",
    ):
        """Add a message to an operation.
        
        Args:
            operation_id: Operation identifier
            message: Message text
            level: Message level (info, warning, error)
        """
        operation = self._operations.get(operation_id)
        if not operation:
            return
        
        event_type = {
            "info": ProgressEventType.MESSAGE,
            "warning": ProgressEventType.WARNING,
            "error": ProgressEventType.ERROR,
        }.get(level, ProgressEventType.MESSAGE)
        
        self._emit_event(ProgressEvent(
            event_id=self._next_event_id(),
            event_type=event_type,
            operation_id=operation_id,
            message=message,
            status=operation.status,
        ))
    
    def start_subtask(
        self,
        operation_id: str,
        subtask_id: str,
        name: str,
        description: str = "",
    ) -> Optional[TrackedOperation]:
        """Start a subtask within an operation.
        
        Args:
            operation_id: Parent operation identifier
            subtask_id: Subtask identifier
            name: Subtask name
            description: Subtask description
            
        Returns:
            The subtask operation
        """
        operation = self._operations.get(operation_id)
        if not operation:
            return None
        
        subtask = TrackedOperation(
            operation_id=subtask_id,
            name=name,
            description=description,
            status=ProgressStatus.INITIALIZING,
            started_at=datetime.now().isoformat(),
        )
        
        operation.subtasks.append(subtask)
        
        self._emit_event(ProgressEvent(
            event_id=self._next_event_id(),
            event_type=ProgressEventType.SUBTASK_STARTED,
            operation_id=operation_id,
            message=f"Subtask started: {name}",
            metadata={"subtask_id": subtask_id, "name": name},
        ))
        
        return subtask
    
    def complete_subtask(
        self,
        operation_id: str,
        subtask_id: str,
        success: bool = True,
        message: str = "",
    ):
        """Complete a subtask.
        
        Args:
            operation_id: Parent operation identifier
            subtask_id: Subtask identifier
            success: Whether subtask succeeded
            message: Completion message
        """
        operation = self._operations.get(operation_id)
        if not operation:
            return
        
        for subtask in operation.subtasks:
            if subtask.operation_id == subtask_id:
                subtask.status = ProgressStatus.COMPLETED if success else ProgressStatus.FAILED
                subtask.completed_at = datetime.now().isoformat()
                subtask.progress_percent = 100.0 if success else subtask.progress_percent
                
                self._emit_event(ProgressEvent(
                    event_id=self._next_event_id(),
                    event_type=ProgressEventType.SUBTASK_COMPLETED,
                    operation_id=operation_id,
                    message=message or f"Subtask completed: {subtask.name}",
                    metadata={"subtask_id": subtask_id, "success": success},
                ))
                break
    
    def complete_operation(
        self,
        operation_id: str,
        success: bool = True,
        message: str = "",
        error_message: Optional[str] = None,
    ):
        """Complete an operation.
        
        Args:
            operation_id: Operation identifier
            success: Whether operation succeeded
            message: Completion message
            error_message: Error message if failed
        """
        operation = self._operations.get(operation_id)
        if not operation:
            return
        
        operation.status = ProgressStatus.COMPLETED if success else ProgressStatus.FAILED
        operation.completed_at = datetime.now().isoformat()
        operation.progress_percent = 100.0 if success else operation.progress_percent
        operation.error_message = error_message
        
        self._emit_event(ProgressEvent(
            event_id=self._next_event_id(),
            event_type=ProgressEventType.COMPLETED,
            operation_id=operation_id,
            message=message or f"Operation {'completed' if success else 'failed'}: {operation.name}",
            progress_percent=100.0 if success else operation.progress_percent,
            status=operation.status,
            metadata={"success": success, "error": error_message},
        ))
        
        # Move to history
        self._operation_history.append(operation)
        # Keep recent operations accessible
        if len(self._operation_history) > 100:
            self._operation_history = self._operation_history[-100:]
    
    def cancel_operation(self, operation_id: str, message: str = ""):
        """Cancel an operation.
        
        Args:
            operation_id: Operation identifier
            message: Cancellation message
        """
        operation = self._operations.get(operation_id)
        if not operation:
            return
        
        operation.status = ProgressStatus.CANCELLED
        operation.completed_at = datetime.now().isoformat()
        
        self._emit_event(ProgressEvent(
            event_id=self._next_event_id(),
            event_type=ProgressEventType.STATUS_CHANGE,
            operation_id=operation_id,
            message=message or f"Operation cancelled: {operation.name}",
            status=ProgressStatus.CANCELLED,
        ))
    
    def get_operation(self, operation_id: str) -> Optional[TrackedOperation]:
        """Get an operation by ID.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            The operation or None
        """
        return self._operations.get(operation_id)
    
    def get_active_operations(self) -> List[TrackedOperation]:
        """Get all active operations.
        
        Returns:
            List of active operations
        """
        return [
            op for op in self._operations.values()
            if op.status in {
                ProgressStatus.INITIALIZING,
                ProgressStatus.IN_PROGRESS,
                ProgressStatus.WAITING,
                ProgressStatus.PAUSED,
            }
        ]
    
    def add_callback(self, callback: Callable[[ProgressEvent], None]):
        """Add an event callback.
        
        Args:
            callback: Function to call for each event
        """
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[ProgressEvent], None]):
        """Remove an event callback.
        
        Args:
            callback: Callback to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _emit_event(self, event: ProgressEvent):
        """Emit an event to all callbacks.
        
        Args:
            event: The event to emit
        """
        # Store in operation
        operation = self._operations.get(event.operation_id)
        if operation:
            operation.events.append(event)
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def _next_event_id(self) -> str:
        """Generate next event ID."""
        self._event_counter += 1
        return f"evt_{self._event_counter}"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracked operations.
        
        Returns:
            Summary dictionary
        """
        active = self.get_active_operations()
        
        return {
            "total_operations": len(self._operations),
            "active_operations": len(active),
            "completed_operations": len([
                op for op in self._operations.values()
                if op.status == ProgressStatus.COMPLETED
            ]),
            "failed_operations": len([
                op for op in self._operations.values()
                if op.status == ProgressStatus.FAILED
            ]),
            "operations": {
                op_id: op.to_dict()
                for op_id, op in self._operations.items()
            },
        }


# Global tracker instance
_global_tracker: Optional[ProgressTracker] = None


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ProgressTracker()
    return _global_tracker
