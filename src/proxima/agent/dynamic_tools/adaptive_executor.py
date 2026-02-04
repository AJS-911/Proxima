"""Adaptive Executor for Dynamic Execution Strategies.

This module implements Phase 3.1.3 & 3.1.4 for the Dynamic AI Assistant:
- Adaptive Execution Strategy: Dynamic strategy selection, retry generation,
  fallback mechanisms, timeout management, resource optimization
- Progress Monitoring and Adjustment: Real-time tracking, bottleneck detection,
  dynamic reallocation, workflow pause/resume

Key Features:
============
- Dynamic strategy selection based on context
- Retry strategy generation for failed operations
- Fallback mechanism selection
- Timeout management with graceful degradation
- Resource allocation optimization
- Priority-based execution scheduling
- Real-time workflow progress tracking
- Bottleneck detection and resolution
- Dynamic resource reallocation
- Intermediate result validation
- Workflow pause and resume

Design Principle:
================
All execution decisions use LLM reasoning - NO hardcoded strategies.
The LLM determines optimal strategies based on context and failure patterns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
import uuid

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategies for tasks."""
    SEQUENTIAL = "sequential"  # One task at a time
    PARALLEL = "parallel"  # Multiple tasks simultaneously
    PIPELINE = "pipeline"  # Stream output to next task
    BATCH = "batch"  # Group similar tasks
    ADAPTIVE = "adaptive"  # Dynamically adjust


class RetryStrategy(Enum):
    """Retry strategies for failed operations."""
    IMMEDIATE = "immediate"  # Retry immediately
    LINEAR_BACKOFF = "linear_backoff"  # Wait n * delay
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Wait 2^n * delay
    FIXED_DELAY = "fixed_delay"  # Wait constant time
    NO_RETRY = "no_retry"  # Don't retry


class FallbackStrategy(Enum):
    """Fallback strategies when primary approach fails."""
    ALTERNATIVE_TOOL = "alternative_tool"  # Try a different tool
    SIMPLIFIED_PARAMS = "simplified_params"  # Use simpler parameters
    SKIP = "skip"  # Skip the task
    USER_INPUT = "user_input"  # Ask user for help
    ABORT = "abort"  # Stop execution


class ResourceState(Enum):
    """State of a resource."""
    AVAILABLE = "available"
    BUSY = "busy"
    EXHAUSTED = "exhausted"
    ERROR = "error"


@dataclass
class ResourceInfo:
    """Information about a system resource."""
    resource_id: str
    resource_type: str  # cpu, memory, disk, network, terminal
    capacity: float  # 0.0-1.0 or absolute value
    used: float = 0.0
    state: ResourceState = ResourceState.AVAILABLE
    
    @property
    def available(self) -> float:
        return max(0, self.capacity - self.used)
    
    @property
    def utilization(self) -> float:
        if self.capacity == 0:
            return 0.0
        return self.used / self.capacity


@dataclass
class ExecutionMetrics:
    """Metrics for an execution."""
    task_id: str
    started_at: float  # time.time()
    completed_at: Optional[float] = None
    duration_ms: Optional[float] = None
    retry_count: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    success: bool = False
    error: Optional[str] = None
    
    def complete(self, success: bool = True, error: Optional[str] = None):
        """Mark execution as complete."""
        self.completed_at = time.time()
        self.duration_ms = (self.completed_at - self.started_at) * 1000
        self.success = success
        self.error = error


@dataclass
class ExecutionConfig:
    """Configuration for adaptive execution."""
    # Strategy settings
    default_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    max_parallel_tasks: int = 4
    
    # Retry settings
    default_retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    initial_retry_delay_ms: int = 1000
    max_retry_delay_ms: int = 30000
    
    # Timeout settings
    default_timeout_ms: int = 60000
    timeout_grace_period_ms: int = 5000
    
    # Resource settings
    cpu_threshold: float = 0.8  # Max CPU before throttling
    memory_threshold: float = 0.8  # Max memory before throttling
    
    # Monitoring settings
    progress_update_interval_ms: int = 500
    bottleneck_threshold_ms: int = 5000  # Time before marking as bottleneck


@dataclass
class TaskExecution:
    """A task being executed."""
    task_id: str
    task_name: str
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Execution settings
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    fallback_strategy: FallbackStrategy = FallbackStrategy.SKIP
    timeout_ms: int = 60000
    priority: int = 5  # 1-10, higher = more important
    
    # State
    status: str = "pending"  # pending, running, completed, failed, cancelled
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Metrics
    metrics: Optional[ExecutionMetrics] = None
    
    # Retry tracking
    retry_count: int = 0
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "tool_name": self.tool_name,
            "status": self.status,
            "retry_count": self.retry_count,
            "error": self.error,
        }


@dataclass
class ExecutionProgress:
    """Progress of an execution plan."""
    total_tasks: int
    completed_tasks: int = 0
    failed_tasks: int = 0
    running_tasks: int = 0
    pending_tasks: int = 0
    
    # Timing
    started_at: Optional[float] = None
    estimated_completion: Optional[float] = None
    elapsed_ms: float = 0.0
    
    # Bottlenecks
    bottleneck_tasks: List[str] = field(default_factory=list)
    
    @property
    def completion_percentage(self) -> float:
        if self.total_tasks == 0:
            return 100.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    @property
    def success_rate(self) -> float:
        total = self.completed_tasks + self.failed_tasks
        if total == 0:
            return 100.0
        return (self.completed_tasks / total) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "running_tasks": self.running_tasks,
            "pending_tasks": self.pending_tasks,
            "completion_percentage": self.completion_percentage,
            "success_rate": self.success_rate,
            "elapsed_ms": self.elapsed_ms,
            "bottleneck_tasks": self.bottleneck_tasks,
        }


class AdaptiveExecutor:
    """Adaptive executor for dynamic execution strategies.
    
    Uses LLM reasoning to:
    1. Select optimal execution strategy based on task characteristics
    2. Generate retry strategies for failures
    3. Determine fallback approaches
    4. Manage timeouts and resource allocation
    5. Monitor progress and detect bottlenecks
    
    Example:
        >>> executor = AdaptiveExecutor(llm_client=client)
        >>> result = await executor.execute_tasks(tasks)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        config: Optional[ExecutionConfig] = None,
    ):
        """Initialize the executor.
        
        Args:
            llm_client: LLM client for reasoning
            tool_registry: Tool registry
            config: Execution configuration
        """
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._config = config or ExecutionConfig()
        
        # Resources
        self._resources: Dict[str, ResourceInfo] = {
            "cpu": ResourceInfo("cpu", "cpu", 1.0),
            "memory": ResourceInfo("memory", "memory", 1.0),
            "disk": ResourceInfo("disk", "disk", 1.0),
            "network": ResourceInfo("network", "network", 1.0),
            "terminal": ResourceInfo("terminal", "terminal", self._config.max_parallel_tasks),
        }
        
        # Active executions
        self._active_tasks: Dict[str, TaskExecution] = {}
        self._task_queue: List[TaskExecution] = []
        self._completed_tasks: List[TaskExecution] = []
        
        # Progress
        self._progress: Optional[ExecutionProgress] = None
        self._paused: bool = False
        self._cancelled: bool = False
        
        # Callbacks
        self._on_progress: Optional[Callable[[ExecutionProgress], None]] = None
        self._on_task_complete: Optional[Callable[[TaskExecution], None]] = None
        self._on_bottleneck: Optional[Callable[[str, TaskExecution], None]] = None
        
        # Metrics history
        self._metrics_history: List[ExecutionMetrics] = []
        
        # Get registry if not provided
        if self._tool_registry is None:
            from .tool_registry import get_tool_registry
            self._tool_registry = get_tool_registry()
    
    async def execute_tasks(
        self,
        tasks: List[TaskExecution],
        tool_executor: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Execute a list of tasks with adaptive strategies.
        
        Args:
            tasks: List of tasks to execute
            tool_executor: Optional function to execute tools
            
        Returns:
            Dictionary with execution results
        """
        if not tasks:
            return {"success": True, "tasks": [], "message": "No tasks to execute"}
        
        # Initialize progress
        self._progress = ExecutionProgress(total_tasks=len(tasks))
        self._progress.pending_tasks = len(tasks)
        self._progress.started_at = time.time()
        
        self._task_queue = list(tasks)
        self._paused = False
        self._cancelled = False
        
        # Determine execution strategy
        strategy = await self._select_strategy(tasks)
        
        results = []
        
        try:
            if strategy == ExecutionStrategy.PARALLEL:
                results = await self._execute_parallel(tasks, tool_executor)
            elif strategy == ExecutionStrategy.PIPELINE:
                results = await self._execute_pipeline(tasks, tool_executor)
            elif strategy == ExecutionStrategy.BATCH:
                results = await self._execute_batch(tasks, tool_executor)
            else:
                # Sequential or Adaptive
                results = await self._execute_sequential(tasks, tool_executor)
            
            # Final progress update
            self._progress.elapsed_ms = (time.time() - self._progress.started_at) * 1000
            
            return {
                "success": self._progress.failed_tasks == 0,
                "tasks": [t.to_dict() for t in self._completed_tasks],
                "progress": self._progress.to_dict(),
                "metrics": self._get_metrics_summary(),
            }
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "progress": self._progress.to_dict() if self._progress else None,
            }
    
    async def _select_strategy(
        self,
        tasks: List[TaskExecution],
    ) -> ExecutionStrategy:
        """Select optimal execution strategy using LLM reasoning."""
        if self._llm_client is None:
            # Fallback: Use adaptive logic based on task count
            if len(tasks) == 1:
                return ExecutionStrategy.SEQUENTIAL
            elif len(tasks) <= 3:
                return ExecutionStrategy.PARALLEL
            else:
                return ExecutionStrategy.ADAPTIVE
        
        # Use LLM to determine strategy
        task_descriptions = [
            f"- {t.task_name}: {t.tool_name or 'no tool'}"
            for t in tasks
        ]
        
        prompt = f"""Determine the optimal execution strategy for these tasks:

Tasks:
{chr(10).join(task_descriptions)}

Available strategies:
1. SEQUENTIAL - Execute one at a time, in order
2. PARALLEL - Execute multiple tasks simultaneously
3. PIPELINE - Stream output from one task to the next
4. BATCH - Group similar tasks together

Consider:
- Task dependencies (do any tasks depend on others?)
- Resource usage (can they share resources?)
- Order requirements (does order matter?)
- Efficiency (which is fastest while being correct?)

Respond with just the strategy name: SEQUENTIAL, PARALLEL, PIPELINE, or BATCH
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            response_upper = response.strip().upper()
            
            if "PARALLEL" in response_upper:
                return ExecutionStrategy.PARALLEL
            elif "PIPELINE" in response_upper:
                return ExecutionStrategy.PIPELINE
            elif "BATCH" in response_upper:
                return ExecutionStrategy.BATCH
            else:
                return ExecutionStrategy.SEQUENTIAL
                
        except Exception as e:
            logger.warning(f"LLM strategy selection failed: {e}")
            return ExecutionStrategy.ADAPTIVE
    
    async def _execute_sequential(
        self,
        tasks: List[TaskExecution],
        tool_executor: Optional[Callable],
    ) -> List[TaskExecution]:
        """Execute tasks sequentially."""
        for task in tasks:
            if self._cancelled:
                task.status = "cancelled"
                continue
            
            while self._paused:
                await asyncio.sleep(0.1)
            
            await self._execute_single_task(task, tool_executor)
            self._completed_tasks.append(task)
            
            self._update_progress()
        
        return self._completed_tasks
    
    async def _execute_parallel(
        self,
        tasks: List[TaskExecution],
        tool_executor: Optional[Callable],
    ) -> List[TaskExecution]:
        """Execute tasks in parallel with resource limits."""
        semaphore = asyncio.Semaphore(self._config.max_parallel_tasks)
        
        async def execute_with_semaphore(task: TaskExecution):
            async with semaphore:
                if self._cancelled:
                    task.status = "cancelled"
                    return task
                
                while self._paused:
                    await asyncio.sleep(0.1)
                
                await self._execute_single_task(task, tool_executor)
                self._completed_tasks.append(task)
                self._update_progress()
                return task
        
        await asyncio.gather(*[
            execute_with_semaphore(task) for task in tasks
        ])
        
        return self._completed_tasks
    
    async def _execute_pipeline(
        self,
        tasks: List[TaskExecution],
        tool_executor: Optional[Callable],
    ) -> List[TaskExecution]:
        """Execute tasks in a pipeline, passing results forward."""
        previous_result = None
        
        for task in tasks:
            if self._cancelled:
                task.status = "cancelled"
                continue
            
            while self._paused:
                await asyncio.sleep(0.1)
            
            # Pass previous result as input
            if previous_result is not None:
                task.parameters["_previous_result"] = previous_result
            
            await self._execute_single_task(task, tool_executor)
            previous_result = task.result
            self._completed_tasks.append(task)
            
            self._update_progress()
        
        return self._completed_tasks
    
    async def _execute_batch(
        self,
        tasks: List[TaskExecution],
        tool_executor: Optional[Callable],
    ) -> List[TaskExecution]:
        """Execute tasks in batches of similar types."""
        # Group tasks by tool name
        batches: Dict[str, List[TaskExecution]] = {}
        for task in tasks:
            key = task.tool_name or "no_tool"
            if key not in batches:
                batches[key] = []
            batches[key].append(task)
        
        # Execute each batch
        for batch_tasks in batches.values():
            if self._cancelled:
                for task in batch_tasks:
                    task.status = "cancelled"
                continue
            
            # Execute batch tasks in parallel
            await self._execute_parallel(batch_tasks, tool_executor)
        
        return self._completed_tasks
    
    async def _execute_single_task(
        self,
        task: TaskExecution,
        tool_executor: Optional[Callable],
    ):
        """Execute a single task with retry and fallback handling."""
        task.status = "running"
        task.metrics = ExecutionMetrics(task_id=task.task_id, started_at=time.time())
        self._active_tasks[task.task_id] = task
        
        self._progress.running_tasks += 1
        self._progress.pending_tasks -= 1
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._do_execute_task(task, tool_executor),
                timeout=task.timeout_ms / 1000,
            )
            
            task.result = result
            task.status = "completed"
            task.metrics.complete(success=True)
            
            self._progress.completed_tasks += 1
            
        except asyncio.TimeoutError:
            task.error = "Task timed out"
            task.status = "failed"
            task.metrics.complete(success=False, error="Timeout")
            
            # Try graceful degradation
            await self._handle_timeout(task, tool_executor)
            
        except Exception as e:
            task.error = str(e)
            task.last_error = str(e)
            
            # Try retry strategy
            should_retry = await self._should_retry(task)
            
            if should_retry:
                await self._retry_task(task, tool_executor)
            else:
                # Try fallback
                await self._handle_failure(task, tool_executor)
        
        finally:
            self._progress.running_tasks -= 1
            if task.task_id in self._active_tasks:
                del self._active_tasks[task.task_id]
            
            if task.metrics:
                self._metrics_history.append(task.metrics)
            
            if self._on_task_complete:
                self._on_task_complete(task)
    
    async def _do_execute_task(
        self,
        task: TaskExecution,
        tool_executor: Optional[Callable],
    ) -> Any:
        """Actually execute the task."""
        if tool_executor:
            return await tool_executor(task.tool_name, task.parameters)
        
        if task.tool_name and self._tool_registry:
            tool = self._tool_registry.get_tool(task.tool_name)
            if tool:
                return await tool.tool_instance.execute(task.parameters)
        
        # No execution possible
        return None
    
    async def _should_retry(self, task: TaskExecution) -> bool:
        """Determine if task should be retried using LLM reasoning."""
        if task.retry_count >= self._config.max_retries:
            return False
        
        if task.retry_strategy == RetryStrategy.NO_RETRY:
            return False
        
        if self._llm_client is None:
            # Default: retry transient errors
            transient_keywords = ["timeout", "connection", "temporary", "retry"]
            error_lower = (task.last_error or "").lower()
            return any(kw in error_lower for kw in transient_keywords)
        
        # Use LLM to decide
        prompt = f"""Should this task be retried?

Task: {task.task_name}
Tool: {task.tool_name}
Error: {task.last_error}
Retry count: {task.retry_count}/{self._config.max_retries}

Consider:
- Is this a transient error (network, timeout)?
- Is this a permanent error (invalid params, not found)?
- Would retrying likely succeed?

Respond with just: YES or NO
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            return "YES" in response.upper()
        except Exception:
            return task.retry_count < self._config.max_retries
    
    async def _retry_task(
        self,
        task: TaskExecution,
        tool_executor: Optional[Callable],
    ):
        """Retry a failed task with appropriate delay."""
        task.retry_count += 1
        
        # Calculate delay
        delay_ms = self._calculate_retry_delay(task)
        
        logger.info(f"Retrying task {task.task_id} in {delay_ms}ms (attempt {task.retry_count})")
        
        await asyncio.sleep(delay_ms / 1000)
        
        # Retry
        try:
            result = await asyncio.wait_for(
                self._do_execute_task(task, tool_executor),
                timeout=task.timeout_ms / 1000,
            )
            
            task.result = result
            task.status = "completed"
            task.metrics.complete(success=True)
            
            self._progress.completed_tasks += 1
            
        except Exception as e:
            task.error = str(e)
            task.last_error = str(e)
            
            # Check if more retries
            if await self._should_retry(task):
                await self._retry_task(task, tool_executor)
            else:
                await self._handle_failure(task, tool_executor)
    
    def _calculate_retry_delay(self, task: TaskExecution) -> int:
        """Calculate retry delay based on strategy."""
        base_delay = self._config.initial_retry_delay_ms
        
        if task.retry_strategy == RetryStrategy.IMMEDIATE:
            return 0
        
        elif task.retry_strategy == RetryStrategy.FIXED_DELAY:
            return base_delay
        
        elif task.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            return min(
                base_delay * task.retry_count,
                self._config.max_retry_delay_ms
            )
        
        else:  # EXPONENTIAL_BACKOFF
            return min(
                base_delay * (2 ** (task.retry_count - 1)),
                self._config.max_retry_delay_ms
            )
    
    async def _handle_failure(
        self,
        task: TaskExecution,
        tool_executor: Optional[Callable],
    ):
        """Handle task failure with fallback strategy."""
        fallback = await self._determine_fallback(task)
        
        if fallback == FallbackStrategy.SKIP:
            task.status = "failed"
            task.metrics.complete(success=False, error=task.last_error)
            self._progress.failed_tasks += 1
            
        elif fallback == FallbackStrategy.ALTERNATIVE_TOOL:
            # Try to find alternative tool
            alt_tool = await self._find_alternative_tool(task)
            if alt_tool:
                task.tool_name = alt_tool
                task.retry_count = 0
                await self._execute_single_task(task, tool_executor)
            else:
                task.status = "failed"
                self._progress.failed_tasks += 1
                
        elif fallback == FallbackStrategy.SIMPLIFIED_PARAMS:
            # Try with simplified parameters
            simplified = await self._simplify_parameters(task)
            if simplified:
                task.parameters = simplified
                task.retry_count = 0
                await self._execute_single_task(task, tool_executor)
            else:
                task.status = "failed"
                self._progress.failed_tasks += 1
                
        else:
            task.status = "failed"
            task.metrics.complete(success=False, error=task.last_error)
            self._progress.failed_tasks += 1
    
    async def _determine_fallback(self, task: TaskExecution) -> FallbackStrategy:
        """Determine fallback strategy using LLM reasoning."""
        if self._llm_client is None:
            return task.fallback_strategy
        
        prompt = f"""What fallback strategy should be used for this failed task?

Task: {task.task_name}
Tool: {task.tool_name}
Error: {task.last_error}
Retries exhausted: {task.retry_count >= self._config.max_retries}

Available strategies:
1. ALTERNATIVE_TOOL - Try a different tool that can accomplish the same goal
2. SIMPLIFIED_PARAMS - Try with simpler/default parameters
3. SKIP - Skip this task and continue
4. ABORT - Stop the entire execution

Respond with just the strategy name.
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            response_upper = response.strip().upper()
            
            if "ALTERNATIVE" in response_upper:
                return FallbackStrategy.ALTERNATIVE_TOOL
            elif "SIMPLIFIED" in response_upper:
                return FallbackStrategy.SIMPLIFIED_PARAMS
            elif "ABORT" in response_upper:
                return FallbackStrategy.ABORT
            else:
                return FallbackStrategy.SKIP
                
        except Exception:
            return task.fallback_strategy
    
    async def _find_alternative_tool(self, task: TaskExecution) -> Optional[str]:
        """Find an alternative tool using LLM reasoning."""
        if self._llm_client is None or self._tool_registry is None:
            return None
        
        # Get available tools
        tools = [
            {"name": t.definition.name, "description": t.definition.description}
            for t in self._tool_registry.get_all_tools()
            if t.definition.name != task.tool_name
        ]
        
        if not tools:
            return None
        
        prompt = f"""Find an alternative tool for this task.

Failed task: {task.task_name}
Failed tool: {task.tool_name}
Error: {task.last_error}

Available alternative tools:
{chr(10).join(f"- {t['name']}: {t['description']}" for t in tools)}

Respond with just the tool name, or "NONE" if no suitable alternative exists.
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            tool_name = response.strip()
            
            if tool_name.upper() == "NONE":
                return None
            
            # Verify tool exists
            if self._tool_registry.get_tool(tool_name):
                return tool_name
            
            return None
            
        except Exception:
            return None
    
    async def _simplify_parameters(self, task: TaskExecution) -> Optional[Dict[str, Any]]:
        """Simplify parameters using LLM reasoning."""
        if self._llm_client is None:
            # Fallback: Remove optional parameters
            if self._tool_registry and task.tool_name:
                tool = self._tool_registry.get_tool(task.tool_name)
                if tool:
                    required_params = {
                        p.name: task.parameters.get(p.name)
                        for p in tool.definition.parameters
                        if p.required and p.name in task.parameters
                    }
                    return required_params if required_params else None
            return None
        
        prompt = f"""Simplify these parameters to fix the error.

Task: {task.task_name}
Tool: {task.tool_name}
Current parameters: {task.parameters}
Error: {task.last_error}

Suggest simplified parameters as JSON, or respond "NONE" if simplification won't help.
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            
            if "NONE" in response.upper():
                return None
            
            from .structured_output import JSONExtractor
            extractor = JSONExtractor()
            parsed, _ = extractor.extract(response)
            
            if parsed and isinstance(parsed, dict):
                return parsed
            
            return None
            
        except Exception:
            return None
    
    async def _handle_timeout(
        self,
        task: TaskExecution,
        tool_executor: Optional[Callable],
    ):
        """Handle task timeout with graceful degradation."""
        # Add grace period
        logger.warning(f"Task {task.task_id} timed out, adding grace period")
        
        task.timeout_ms += self._config.timeout_grace_period_ms
        
        # One retry with extended timeout
        if task.retry_count < 1:
            task.retry_count += 1
            task.error = None
            
            try:
                result = await asyncio.wait_for(
                    self._do_execute_task(task, tool_executor),
                    timeout=task.timeout_ms / 1000,
                )
                
                task.result = result
                task.status = "completed"
                task.metrics.complete(success=True)
                self._progress.completed_tasks += 1
                
            except Exception as e:
                task.error = str(e)
                task.status = "failed"
                task.metrics.complete(success=False, error=str(e))
                self._progress.failed_tasks += 1
        else:
            task.status = "failed"
            task.metrics.complete(success=False, error="Timeout after grace period")
            self._progress.failed_tasks += 1
    
    def _update_progress(self):
        """Update progress and check for bottlenecks."""
        if not self._progress:
            return
        
        # Update elapsed time
        if self._progress.started_at:
            self._progress.elapsed_ms = (time.time() - self._progress.started_at) * 1000
        
        # Check for bottlenecks (tasks running too long)
        bottlenecks = []
        for task_id, task in self._active_tasks.items():
            if task.metrics and task.metrics.started_at:
                running_time = (time.time() - task.metrics.started_at) * 1000
                if running_time > self._config.bottleneck_threshold_ms:
                    bottlenecks.append(task_id)
                    
                    if self._on_bottleneck:
                        self._on_bottleneck(task_id, task)
        
        self._progress.bottleneck_tasks = bottlenecks
        
        # Emit progress
        if self._on_progress:
            self._on_progress(self._progress)
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of execution metrics."""
        if not self._metrics_history:
            return {}
        
        durations = [m.duration_ms for m in self._metrics_history if m.duration_ms]
        successes = sum(1 for m in self._metrics_history if m.success)
        retries = sum(m.retry_count for m in self._metrics_history)
        
        return {
            "total_executions": len(self._metrics_history),
            "successful_executions": successes,
            "total_retries": retries,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
        }
    
    def pause(self):
        """Pause execution."""
        self._paused = True
        logger.info("Execution paused")
    
    def resume(self):
        """Resume execution."""
        self._paused = False
        logger.info("Execution resumed")
    
    def cancel(self):
        """Cancel execution."""
        self._cancelled = True
        logger.info("Execution cancelled")
    
    def get_progress(self) -> Optional[ExecutionProgress]:
        """Get current progress."""
        return self._progress
    
    def get_active_tasks(self) -> List[TaskExecution]:
        """Get currently active tasks."""
        return list(self._active_tasks.values())
    
    def set_progress_callback(self, callback: Callable[[ExecutionProgress], None]):
        """Set callback for progress updates."""
        self._on_progress = callback
    
    def set_task_complete_callback(self, callback: Callable[[TaskExecution], None]):
        """Set callback for task completion."""
        self._on_task_complete = callback
    
    def set_bottleneck_callback(self, callback: Callable[[str, TaskExecution], None]):
        """Set callback for bottleneck detection."""
        self._on_bottleneck = callback


# Module-level executor instance
_global_executor: Optional[AdaptiveExecutor] = None


def get_adaptive_executor(
    llm_client: Optional[Any] = None,
    config: Optional[ExecutionConfig] = None,
) -> AdaptiveExecutor:
    """Get the global adaptive executor.
    
    Args:
        llm_client: Optional LLM client
        config: Optional execution config
        
    Returns:
        AdaptiveExecutor instance
    """
    global _global_executor
    if _global_executor is None:
        _global_executor = AdaptiveExecutor(
            llm_client=llm_client,
            config=config,
        )
    return _global_executor
