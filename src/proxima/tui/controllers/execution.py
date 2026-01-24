"""Execution Controller for Proxima TUI.

Handles execution management, progress tracking, and control signals.
"""

from typing import Optional, Callable, List, Dict, Any
from datetime import datetime
from enum import Enum, auto

from ..state import TUIState
from ..state.tui_state import StageInfo, CheckpointInfo
from ..state.events import (
    ExecutionStarted,
    ExecutionProgress,
    ExecutionCompleted,
    ExecutionFailed,
    ExecutionPaused,
    ExecutionResumed,
    ExecutionAborted,
    StageStarted,
    StageCompleted,
    CheckpointCreated,
    RollbackCompleted,
)


class ControlSignal(Enum):
    """Control signals for execution."""
    NONE = auto()
    START = auto()
    PAUSE = auto()
    RESUME = auto()
    ABORT = auto()
    ROLLBACK = auto()


class ExecutionController:
    """Controller for execution management.
    
    Handles starting, pausing, resuming, and aborting executions,
    as well as tracking progress and managing checkpoints.
    """
    
    def __init__(self, state: TUIState):
        """Initialize the execution controller.
        
        Args:
            state: The TUI state instance
        """
        self.state = state
        self._executor = None  # Will be set when Proxima core is available
        self._control = None   # Will be set when Proxima core is available
        self._event_callbacks: List[Callable] = []
    
    @property
    def is_running(self) -> bool:
        """Check if execution is running."""
        return self.state.execution_status == "RUNNING"
    
    @property
    def is_paused(self) -> bool:
        """Check if execution is paused."""
        return self.state.execution_status == "PAUSED"
    
    @property
    def is_idle(self) -> bool:
        """Check if no execution is active."""
        return self.state.execution_status == "IDLE"
    
    @property
    def can_pause(self) -> bool:
        """Check if execution can be paused."""
        return self.is_running
    
    @property
    def can_resume(self) -> bool:
        """Check if execution can be resumed."""
        return self.is_paused
    
    @property
    def can_abort(self) -> bool:
        """Check if execution can be aborted."""
        return self.is_running or self.is_paused
    
    @property
    def can_rollback(self) -> bool:
        """Check if rollback is available."""
        return self.state.rollback_available and self.state.checkpoint_count > 0
    
    def start_execution(
        self,
        task: str,
        backend: str,
        simulator: str = "statevector",
        qubits: int = 2,
        shots: int = 1024,
        **config,
    ) -> bool:
        """Start a new execution.
        
        Args:
            task: Task/circuit name
            backend: Backend name
            simulator: Simulator type
            qubits: Number of qubits
            shots: Number of shots
            **config: Additional configuration
        
        Returns:
            True if execution started successfully
        """
        if not self.is_idle:
            return False
        
        # Update state
        self.state.execution_status = "PLANNING"
        self.state.current_task = task
        self.state.current_task_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state.current_backend = backend
        self.state.current_simulator = simulator
        self.state.qubits = qubits
        self.state.shots = shots
        self.state.progress_percent = 0.0
        self.state.elapsed_ms = 0.0
        self.state.eta_ms = None
        
        # Initialize stages
        self.state.all_stages = [
            StageInfo(name="Planning", index=0),
            StageInfo(name="Backend Initialization", index=1),
            StageInfo(name="Simulation", index=2),
            StageInfo(name="Analysis", index=3),
            StageInfo(name="Report Generation", index=4),
        ]
        self.state.total_stages = len(self.state.all_stages)
        self.state.stage_index = 0
        self.state.current_stage = "Planning"
        
        # Emit event
        self._emit_event(ExecutionStarted(
            task=task,
            backend=backend,
            simulator=simulator,
            session_id=self.state.active_session_id or "default",
            qubits=qubits,
            shots=shots,
        ))
        
        # TODO: Actually start execution via Proxima core
        # self._executor.run(plan)
        
        return True
    
    def pause(self) -> bool:
        """Pause current execution.
        
        Returns:
            True if pause was successful
        """
        if not self.can_pause:
            return False
        
        self.state.execution_status = "PAUSED"
        
        # Create checkpoint
        checkpoint = CheckpointInfo(
            id=f"cp_{datetime.now().strftime('%H%M%S')}",
            stage_index=self.state.stage_index,
            timestamp=datetime.now(),
        )
        self.state.latest_checkpoint = checkpoint
        self.state.checkpoint_count += 1
        self.state.last_checkpoint_time = datetime.now()
        self.state.rollback_available = True
        
        self._emit_event(ExecutionPaused(
            checkpoint_id=checkpoint.id,
            stage_index=checkpoint.stage_index,
        ))
        
        # TODO: Send pause signal to Proxima core
        # self._control.signal(ControlSignal.PAUSE)
        
        return True
    
    def resume(self) -> bool:
        """Resume paused execution.
        
        Returns:
            True if resume was successful
        """
        if not self.can_resume:
            return False
        
        self.state.execution_status = "RUNNING"
        
        checkpoint_id = self.state.latest_checkpoint.id if self.state.latest_checkpoint else "unknown"
        
        self._emit_event(ExecutionResumed(
            checkpoint_id=checkpoint_id,
            stage_index=self.state.stage_index,
        ))
        
        # TODO: Send resume signal to Proxima core
        # self._control.signal(ControlSignal.RESUME)
        
        return True
    
    def abort(self, reason: str = "User requested") -> bool:
        """Abort current execution.
        
        Args:
            reason: Reason for abort
        
        Returns:
            True if abort was successful
        """
        if not self.can_abort:
            return False
        
        self.state.execution_status = "ABORTED"
        
        self._emit_event(ExecutionAborted(reason=reason))
        
        # TODO: Send abort signal to Proxima core
        # self._control.signal(ControlSignal.ABORT, reason=reason)
        
        # Clear execution state after short delay
        self._cleanup_execution()
        
        return True
    
    def rollback(self, checkpoint_id: Optional[str] = None) -> bool:
        """Rollback to a checkpoint.
        
        Args:
            checkpoint_id: Specific checkpoint ID (uses latest if None)
        
        Returns:
            True if rollback was successful
        """
        if not self.can_rollback:
            return False
        
        checkpoint = self.state.latest_checkpoint
        if checkpoint is None:
            return False
        
        # Update state to checkpoint's stage
        self.state.stage_index = checkpoint.stage_index
        self.state.current_stage = self.state.all_stages[checkpoint.stage_index].name
        
        # Mark stages after checkpoint as pending
        for stage in self.state.all_stages[checkpoint.stage_index + 1:]:
            stage.status = "pending"
        
        self._emit_event(RollbackCompleted(
            checkpoint_id=checkpoint.id,
            stage_index=checkpoint.stage_index,
        ))
        
        # TODO: Send rollback signal to Proxima core
        # self._control.signal(ControlSignal.ROLLBACK, checkpoint_id=checkpoint_id)
        
        return True
    
    def update_progress(
        self,
        percent: float,
        stage: str,
        stage_index: int,
        elapsed_ms: float,
        eta_ms: Optional[float] = None,
    ) -> None:
        """Update execution progress.
        
        Args:
            percent: Progress percentage (0-100)
            stage: Current stage name
            stage_index: Current stage index
            elapsed_ms: Elapsed time in milliseconds
            eta_ms: Estimated time remaining in milliseconds
        """
        self.state.update_progress(
            percent=percent,
            stage=stage,
            stage_index=stage_index,
            total_stages=self.state.total_stages,
            elapsed_ms=elapsed_ms,
            eta_ms=eta_ms,
        )
        
        self._emit_event(ExecutionProgress(
            progress=percent,
            stage=stage,
            stage_index=stage_index,
            total_stages=self.state.total_stages,
            elapsed_ms=elapsed_ms,
            eta_ms=eta_ms,
        ))
    
    def complete_stage(self, stage_index: int, duration_ms: float) -> None:
        """Mark a stage as completed.
        
        Args:
            stage_index: Index of the completed stage
            duration_ms: Stage duration in milliseconds
        """
        if stage_index < len(self.state.all_stages):
            stage = self.state.all_stages[stage_index]
            stage.status = "done"
            stage.duration_ms = duration_ms
            stage.end_time = datetime.now()
            
            self.state.completed_stages.append(stage)
            
            self._emit_event(StageCompleted(
                stage_name=stage.name,
                stage_index=stage_index,
                duration_ms=duration_ms,
                success=True,
            ))
    
    def complete_execution(self, result: Dict[str, Any]) -> None:
        """Mark execution as completed.
        
        Args:
            result: Execution result data
        """
        self.state.execution_status = "COMPLETED"
        
        self._emit_event(ExecutionCompleted(
            result=result,
            total_time_ms=self.state.elapsed_ms,
        ))
        
        # Clear execution state after short delay
        self._cleanup_execution()
    
    def fail_execution(self, error: str, stage: str) -> None:
        """Mark execution as failed.
        
        Args:
            error: Error message
            stage: Stage where error occurred
        """
        self.state.execution_status = "ERROR"
        
        self._emit_event(ExecutionFailed(
            error=error,
            stage=stage,
            partial_result=None,
        ))
        
        # Clear execution state after short delay
        self._cleanup_execution()
    
    def _cleanup_execution(self) -> None:
        """Clean up execution state."""
        # Don't clear immediately - let the UI show final state
        pass
    
    def reset(self) -> None:
        """Reset execution state."""
        self.state.clear_execution()
    
    def on_event(self, callback: Callable) -> None:
        """Register an event callback.
        
        Args:
            callback: Function to call on events
        """
        self._event_callbacks.append(callback)
    
    def _emit_event(self, event: Any) -> None:
        """Emit an event to callbacks.
        
        Args:
            event: Event to emit
        """
        for callback in self._event_callbacks:
            callback(event)
