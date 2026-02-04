"""Tool Orchestrator for Dynamic Tool System.

This module provides the orchestration layer that coordinates
multi-tool execution with:
- Execution planning
- Dependency resolution
- Parallel execution where possible
- Error handling and recovery
- Progress tracking

The orchestrator works with the LLM to determine execution strategy
and handles the complexity of multi-step operations.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .tool_interface import (
    ToolInterface,
    ToolResult,
    ToolDefinition,
    RiskLevel,
)
from .tool_registry import ToolRegistry, get_tool_registry
from .execution_context import ExecutionContext, get_current_context

logger = logging.getLogger(__name__)


class ExecutionStepStatus(Enum):
    """Status of an execution step."""
    PENDING = "pending"
    WAITING_DEPENDENCY = "waiting_dependency"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class ExecutionPlanStatus(Enum):
    """Status of an execution plan."""
    DRAFT = "draft"
    READY = "ready"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionStep:
    """A single step in an execution plan."""
    step_id: str
    tool_name: str
    parameters: Dict[str, Any]
    description: str
    dependencies: List[str] = field(default_factory=list)  # step_ids this depends on
    status: ExecutionStepStatus = ExecutionStepStatus.PENDING
    result: Optional[ToolResult] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "description": self.description,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
        }
    
    def can_execute(self, completed_steps: Set[str]) -> bool:
        """Check if this step can execute (all dependencies met)."""
        return all(dep in completed_steps for dep in self.dependencies)


@dataclass
class ExecutionPlan:
    """A plan for executing multiple tools."""
    plan_id: str
    name: str
    description: str
    steps: List[ExecutionStep]
    context: ExecutionContext
    status: ExecutionPlanStatus = ExecutionPlanStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    requires_confirmation: bool = True
    risk_level: RiskLevel = RiskLevel.LOW
    
    def __post_init__(self):
        """Calculate overall risk level."""
        registry = get_tool_registry()
        max_risk = RiskLevel.NONE
        risk_order = list(RiskLevel)
        
        for step in self.steps:
            tool = registry.get_tool(step.tool_name)
            if tool:
                tool_risk = tool.definition.risk_level
                if risk_order.index(tool_risk) > risk_order.index(max_risk):
                    max_risk = tool_risk
        
        self.risk_level = max_risk
        
        # High risk plans require confirmation
        if risk_order.index(max_risk) >= risk_order.index(RiskLevel.HIGH):
            self.requires_confirmation = True
    
    def get_ready_steps(self) -> List[ExecutionStep]:
        """Get steps that are ready to execute."""
        completed = {
            s.step_id for s in self.steps 
            if s.status == ExecutionStepStatus.COMPLETED
        }
        
        ready = []
        for step in self.steps:
            if step.status == ExecutionStepStatus.PENDING:
                if step.can_execute(completed):
                    step.status = ExecutionStepStatus.READY
                    ready.append(step)
        
        return ready
    
    def get_step(self, step_id: str) -> Optional[ExecutionStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "requires_confirmation": self.requires_confirmation,
            "risk_level": self.risk_level.value,
        }
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the plan."""
        lines = [
            f"**Execution Plan: {self.name}**",
            f"Description: {self.description}",
            f"Risk Level: {self.risk_level.value}",
            f"Steps: {len(self.steps)}",
            "",
        ]
        
        for i, step in enumerate(self.steps, 1):
            status_icon = {
                ExecutionStepStatus.PENDING: "⏳",
                ExecutionStepStatus.READY: "🟡",
                ExecutionStepStatus.RUNNING: "🔄",
                ExecutionStepStatus.COMPLETED: "✅",
                ExecutionStepStatus.FAILED: "❌",
                ExecutionStepStatus.SKIPPED: "⏭️",
                ExecutionStepStatus.CANCELLED: "🚫",
            }.get(step.status, "❓")
            
            lines.append(f"{i}. {status_icon} {step.description}")
            lines.append(f"   Tool: {step.tool_name}")
            if step.dependencies:
                lines.append(f"   Depends on: {', '.join(step.dependencies)}")
        
        return "\n".join(lines)


@dataclass
class OrchestratorConfig:
    """Configuration for the tool orchestrator."""
    max_parallel_executions: int = 3
    default_timeout_seconds: float = 60.0
    enable_auto_retry: bool = True
    max_retries: int = 2
    require_confirmation_for_destructive: bool = True
    log_all_executions: bool = True


class ToolOrchestrator:
    """Orchestrates execution of multiple tools.
    
    The orchestrator provides:
    - Execution plan creation and management
    - Dependency-aware execution ordering
    - Parallel execution where safe
    - Error handling and recovery
    - Progress tracking and callbacks
    """
    
    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        config: Optional[OrchestratorConfig] = None,
    ):
        """Initialize the orchestrator.
        
        Args:
            registry: Tool registry to use
            config: Orchestrator configuration
        """
        self._registry = registry or get_tool_registry()
        self._config = config or OrchestratorConfig()
        self._active_plans: Dict[str, ExecutionPlan] = {}
        self._execution_history: List[ExecutionPlan] = []
        self._progress_callbacks: List[Callable[[ExecutionStep, ExecutionPlan], None]] = []
        self._executor = ThreadPoolExecutor(max_workers=self._config.max_parallel_executions)
    
    def create_plan(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        context: Optional[ExecutionContext] = None,
    ) -> ExecutionPlan:
        """Create an execution plan.
        
        Args:
            name: Plan name
            description: Plan description
            steps: List of step definitions with:
                - tool_name: Name of the tool
                - parameters: Tool parameters
                - description: Step description
                - dependencies: Optional list of step IDs this depends on
            context: Execution context (uses current if not provided)
            
        Returns:
            The created execution plan
        """
        context = context or get_current_context()
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        execution_steps = []
        for i, step_def in enumerate(steps):
            step_id = step_def.get("step_id", f"step_{i + 1}")
            execution_steps.append(ExecutionStep(
                step_id=step_id,
                tool_name=step_def["tool_name"],
                parameters=step_def.get("parameters", {}),
                description=step_def.get("description", f"Execute {step_def['tool_name']}"),
                dependencies=step_def.get("dependencies", []),
                max_retries=step_def.get("max_retries", self._config.max_retries),
            ))
        
        plan = ExecutionPlan(
            plan_id=plan_id,
            name=name,
            description=description,
            steps=execution_steps,
            context=context,
            status=ExecutionPlanStatus.READY,
        )
        
        self._active_plans[plan_id] = plan
        return plan
    
    def execute_plan(
        self,
        plan: ExecutionPlan,
        confirm_callback: Optional[Callable[[ExecutionPlan], bool]] = None,
    ) -> ExecutionPlan:
        """Execute a plan synchronously.
        
        Args:
            plan: The plan to execute
            confirm_callback: Optional callback for confirmation
            
        Returns:
            The executed plan with results
        """
        # Check if confirmation needed
        if plan.requires_confirmation and confirm_callback:
            if not confirm_callback(plan):
                plan.status = ExecutionPlanStatus.CANCELLED
                return plan
        
        plan.status = ExecutionPlanStatus.EXECUTING
        plan.started_at = datetime.now().isoformat()
        
        completed_steps: Set[str] = set()
        
        try:
            while True:
                # Get steps that are ready to execute
                ready_steps = plan.get_ready_steps()
                
                if not ready_steps:
                    # Check if all steps are complete or if we're stuck
                    remaining = [
                        s for s in plan.steps 
                        if s.status not in {
                            ExecutionStepStatus.COMPLETED,
                            ExecutionStepStatus.SKIPPED,
                            ExecutionStepStatus.CANCELLED,
                        }
                    ]
                    
                    if not remaining:
                        # All done
                        plan.status = ExecutionPlanStatus.COMPLETED
                        break
                    
                    # Check for failures
                    failed = [s for s in plan.steps if s.status == ExecutionStepStatus.FAILED]
                    if failed:
                        plan.status = ExecutionPlanStatus.FAILED
                        break
                    
                    # Something is wrong - circular dependency or bug
                    logger.error("Execution stuck - possible circular dependency")
                    plan.status = ExecutionPlanStatus.FAILED
                    break
                
                # Execute ready steps (could be parallel)
                for step in ready_steps:
                    self._execute_step(step, plan.context)
                    self._notify_progress(step, plan)
                    
                    if step.status == ExecutionStepStatus.COMPLETED:
                        completed_steps.add(step.step_id)
                    elif step.status == ExecutionStepStatus.FAILED:
                        # Check if we should skip dependent steps
                        self._handle_step_failure(step, plan)
        
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            plan.status = ExecutionPlanStatus.FAILED
        
        finally:
            plan.completed_at = datetime.now().isoformat()
            self._execution_history.append(plan)
        
        return plan
    
    def _execute_step(
        self, 
        step: ExecutionStep, 
        context: ExecutionContext
    ):
        """Execute a single step.
        
        Args:
            step: The step to execute
            context: Execution context
        """
        step.status = ExecutionStepStatus.RUNNING
        step.started_at = datetime.now().isoformat()
        
        try:
            # Get tool instance
            tool = self._registry.get_tool_instance(step.tool_name)
            if not tool:
                raise ValueError(f"Tool not found: {step.tool_name}")
            
            # Validate parameters
            validation = tool.validate_parameters(step.parameters)
            if not validation["valid"]:
                raise ValueError(f"Invalid parameters: {validation['errors']}")
            
            # Execute
            result = tool.execute(step.parameters, context)
            step.result = result
            
            # Update registry usage stats
            self._registry.record_usage(step.tool_name, result.success)
            
            if result.success:
                step.status = ExecutionStepStatus.COMPLETED
            else:
                # Retry logic
                if step.retry_count < step.max_retries and self._config.enable_auto_retry:
                    step.retry_count += 1
                    step.status = ExecutionStepStatus.PENDING
                    logger.info(f"Retrying step {step.step_id} (attempt {step.retry_count})")
                else:
                    step.status = ExecutionStepStatus.FAILED
                    step.error_message = result.error
        
        except Exception as e:
            step.status = ExecutionStepStatus.FAILED
            step.error_message = str(e)
            step.result = ToolResult(
                success=False,
                error=str(e),
            )
            logger.error(f"Step {step.step_id} failed: {e}")
        
        finally:
            step.completed_at = datetime.now().isoformat()
            
            # Record in context
            context.add_execution_record(
                tool_name=step.tool_name,
                parameters=step.parameters,
                result=step.result.to_dict() if step.result else {},
            )
    
    def _handle_step_failure(self, failed_step: ExecutionStep, plan: ExecutionPlan):
        """Handle a step failure by skipping dependent steps.
        
        Args:
            failed_step: The step that failed
            plan: The execution plan
        """
        # Find steps that depend on the failed step
        for step in plan.steps:
            if failed_step.step_id in step.dependencies:
                if step.status in {ExecutionStepStatus.PENDING, ExecutionStepStatus.READY}:
                    step.status = ExecutionStepStatus.SKIPPED
                    step.error_message = f"Skipped due to failure of {failed_step.step_id}"
    
    def _notify_progress(self, step: ExecutionStep, plan: ExecutionPlan):
        """Notify progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(step, plan)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def add_progress_callback(
        self, 
        callback: Callable[[ExecutionStep, ExecutionPlan], None]
    ):
        """Add a progress callback.
        
        Args:
            callback: Function called with (step, plan) on each step completion
        """
        self._progress_callbacks.append(callback)
    
    def execute_single(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[ExecutionContext] = None,
    ) -> ToolResult:
        """Execute a single tool directly.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            context: Execution context
            
        Returns:
            Tool execution result
        """
        context = context or get_current_context()
        
        tool = self._registry.get_tool_instance(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool not found: {tool_name}",
            )
        
        try:
            result = tool.execute(parameters, context)
            self._registry.record_usage(tool_name, result.success)
            
            context.add_execution_record(
                tool_name=tool_name,
                parameters=parameters,
                result=result.to_dict(),
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
            )
    
    async def execute_plan_async(
        self,
        plan: ExecutionPlan,
        confirm_callback: Optional[Callable[[ExecutionPlan], bool]] = None,
    ) -> ExecutionPlan:
        """Execute a plan asynchronously with parallel steps.
        
        Args:
            plan: The plan to execute
            confirm_callback: Optional confirmation callback
            
        Returns:
            The executed plan
        """
        if plan.requires_confirmation and confirm_callback:
            if not confirm_callback(plan):
                plan.status = ExecutionPlanStatus.CANCELLED
                return plan
        
        plan.status = ExecutionPlanStatus.EXECUTING
        plan.started_at = datetime.now().isoformat()
        
        completed_steps: Set[str] = set()
        
        try:
            while True:
                ready_steps = plan.get_ready_steps()
                
                if not ready_steps:
                    remaining = [
                        s for s in plan.steps
                        if s.status not in {
                            ExecutionStepStatus.COMPLETED,
                            ExecutionStepStatus.SKIPPED,
                            ExecutionStepStatus.CANCELLED,
                        }
                    ]
                    
                    if not remaining:
                        plan.status = ExecutionPlanStatus.COMPLETED
                        break
                    
                    failed = [s for s in plan.steps if s.status == ExecutionStepStatus.FAILED]
                    if failed:
                        plan.status = ExecutionPlanStatus.FAILED
                        break
                    
                    plan.status = ExecutionPlanStatus.FAILED
                    break
                
                # Execute ready steps in parallel
                tasks = [
                    self._execute_step_async(step, plan.context)
                    for step in ready_steps
                ]
                await asyncio.gather(*tasks)
                
                for step in ready_steps:
                    self._notify_progress(step, plan)
                    if step.status == ExecutionStepStatus.COMPLETED:
                        completed_steps.add(step.step_id)
                    elif step.status == ExecutionStepStatus.FAILED:
                        self._handle_step_failure(step, plan)
        
        except Exception as e:
            logger.error(f"Async plan execution failed: {e}")
            plan.status = ExecutionPlanStatus.FAILED
        
        finally:
            plan.completed_at = datetime.now().isoformat()
            self._execution_history.append(plan)
        
        return plan
    
    async def _execute_step_async(
        self,
        step: ExecutionStep,
        context: ExecutionContext
    ):
        """Execute a step asynchronously.
        
        Args:
            step: The step to execute
            context: Execution context
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: self._execute_step(step, context)
        )
    
    def pause_plan(self, plan_id: str) -> bool:
        """Pause a running plan.
        
        Args:
            plan_id: ID of the plan to pause
            
        Returns:
            True if paused successfully
        """
        plan = self._active_plans.get(plan_id)
        if plan and plan.status == ExecutionPlanStatus.EXECUTING:
            plan.status = ExecutionPlanStatus.PAUSED
            return True
        return False
    
    def cancel_plan(self, plan_id: str) -> bool:
        """Cancel a plan.
        
        Args:
            plan_id: ID of the plan to cancel
            
        Returns:
            True if cancelled successfully
        """
        plan = self._active_plans.get(plan_id)
        if plan and plan.status in {
            ExecutionPlanStatus.READY,
            ExecutionPlanStatus.EXECUTING,
            ExecutionPlanStatus.PAUSED,
        }:
            plan.status = ExecutionPlanStatus.CANCELLED
            # Mark pending steps as cancelled
            for step in plan.steps:
                if step.status in {
                    ExecutionStepStatus.PENDING,
                    ExecutionStepStatus.READY,
                    ExecutionStepStatus.WAITING_DEPENDENCY,
                }:
                    step.status = ExecutionStepStatus.CANCELLED
            return True
        return False
    
    def get_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        """Get a plan by ID.
        
        Args:
            plan_id: ID of the plan
            
        Returns:
            The plan or None
        """
        return self._active_plans.get(plan_id)
    
    def get_execution_history(self) -> List[ExecutionPlan]:
        """Get execution history.
        
        Returns:
            List of executed plans
        """
        return list(self._execution_history)
    
    def cleanup(self):
        """Cleanup resources."""
        self._executor.shutdown(wait=False)


# Global orchestrator instance
_global_orchestrator: Optional[ToolOrchestrator] = None


def get_tool_orchestrator() -> ToolOrchestrator:
    """Get the global tool orchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = ToolOrchestrator()
    return _global_orchestrator
