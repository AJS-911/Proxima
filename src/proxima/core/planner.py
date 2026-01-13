"""Execution planner implementation with DAG-based task planning.

Planner delegates the reasoning step to an injected callable (LLM or local
model). It drives the execution state machine through planning states and
produces a Directed Acyclic Graph (DAG) of tasks for optimal parallel execution.
"""

from __future__ import annotations

import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from proxima.core.state import ExecutionStateMachine
from proxima.utils.logging import get_logger

PlanFunction = Callable[[str], dict[str, Any]]


class TaskStatus(Enum):
    """Status of a task in the execution DAG."""
    
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class TaskNode:
    """A node in the execution DAG representing a single task.
    
    Attributes:
        task_id: Unique identifier for the task.
        action: The action to perform (e.g., 'create_circuit', 'execute').
        description: Human-readable description.
        parameters: Task-specific parameters.
        dependencies: List of task IDs this task depends on.
        dependents: List of task IDs that depend on this task.
        status: Current execution status.
        result: Task result after execution.
        priority: Execution priority (higher = earlier).
        estimated_duration_ms: Estimated execution time.
        retry_count: Number of retries allowed.
        timeout_s: Timeout in seconds.
        tags: Optional tags for categorization.
    """
    
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    action: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    priority: int = 0
    estimated_duration_ms: float = 100.0
    retry_count: int = 0
    timeout_s: float | None = None
    tags: list[str] = field(default_factory=list)

    def is_ready(self) -> bool:
        """Check if task is ready to execute (all dependencies completed)."""
        return self.status == TaskStatus.READY

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "action": self.action,
            "description": self.description,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "dependents": self.dependents,
            "status": self.status.name,
            "priority": self.priority,
            "estimated_duration_ms": self.estimated_duration_ms,
            "tags": self.tags,
        }


@dataclass
class ExecutionDAG:
    """Directed Acyclic Graph for task execution.
    
    Manages task dependencies and determines execution order for
    optimal parallel execution.
    """
    
    nodes: dict[str, TaskNode] = field(default_factory=dict)
    root_tasks: list[str] = field(default_factory=list)
    _completed_tasks: set[str] = field(default_factory=set)

    def add_task(self, task: TaskNode) -> None:
        """Add a task to the DAG.
        
        Args:
            task: The task node to add.
        """
        self.nodes[task.task_id] = task
        
        # Track root tasks (no dependencies)
        if not task.dependencies:
            self.root_tasks.append(task.task_id)
        
        # Update dependents of dependencies
        for dep_id in task.dependencies:
            if dep_id in self.nodes:
                if task.task_id not in self.nodes[dep_id].dependents:
                    self.nodes[dep_id].dependents.append(task.task_id)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate the DAG for cycles and missing dependencies.
        
        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors: list[str] = []
        
        # Check for missing dependencies
        for task_id, task in self.nodes.items():
            for dep_id in task.dependencies:
                if dep_id not in self.nodes:
                    errors.append(
                        f"Task {task_id} depends on unknown task {dep_id}"
                    )
        
        # Check for cycles using DFS
        visited: set[str] = set()
        rec_stack: set[str] = set()
        
        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = self.nodes.get(node_id)
            if node:
                for dependent_id in node.dependents:
                    if dependent_id not in visited:
                        if has_cycle(dependent_id):
                            return True
                    elif dependent_id in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for task_id in self.nodes:
            if task_id not in visited:
                if has_cycle(task_id):
                    errors.append(f"Cycle detected involving task {task_id}")
                    break
        
        return len(errors) == 0, errors

    def get_ready_tasks(self) -> list[TaskNode]:
        """Get all tasks that are ready to execute.
        
        Returns:
            List of tasks with all dependencies completed.
        """
        ready: list[TaskNode] = []
        
        for task in self.nodes.values():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            all_deps_done = all(
                self.nodes.get(dep_id, TaskNode()).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            
            if all_deps_done:
                ready.append(task)
        
        # Sort by priority (higher first)
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready

    def mark_ready(self, task_id: str) -> None:
        """Mark a task as ready for execution."""
        if task_id in self.nodes:
            self.nodes[task_id].status = TaskStatus.READY

    def mark_running(self, task_id: str) -> None:
        """Mark a task as currently running."""
        if task_id in self.nodes:
            self.nodes[task_id].status = TaskStatus.RUNNING

    def mark_completed(self, task_id: str, result: Any = None) -> list[str]:
        """Mark a task as completed and return newly ready tasks.
        
        Args:
            task_id: The completed task ID.
            result: The task result.
            
        Returns:
            List of task IDs that are now ready.
        """
        if task_id not in self.nodes:
            return []
        
        task = self.nodes[task_id]
        task.status = TaskStatus.COMPLETED
        task.result = result
        self._completed_tasks.add(task_id)
        
        # Find newly ready tasks
        newly_ready: list[str] = []
        for dependent_id in task.dependents:
            dependent = self.nodes.get(dependent_id)
            if dependent and dependent.status == TaskStatus.PENDING:
                # Check if all dependencies are now completed
                all_deps_done = all(
                    self.nodes.get(dep_id, TaskNode()).status == TaskStatus.COMPLETED
                    for dep_id in dependent.dependencies
                )
                if all_deps_done:
                    newly_ready.append(dependent_id)
        
        return newly_ready

    def mark_failed(self, task_id: str, error: str) -> list[str]:
        """Mark a task as failed and return tasks to skip.
        
        Args:
            task_id: The failed task ID.
            error: Error message.
            
        Returns:
            List of dependent task IDs that should be skipped.
        """
        if task_id not in self.nodes:
            return []
        
        task = self.nodes[task_id]
        task.status = TaskStatus.FAILED
        task.error = error
        
        # Find all downstream tasks to skip
        to_skip: list[str] = []
        queue = deque(task.dependents)
        
        while queue:
            dep_id = queue.popleft()
            if dep_id in to_skip:
                continue
            
            dep_task = self.nodes.get(dep_id)
            if dep_task:
                dep_task.status = TaskStatus.SKIPPED
                to_skip.append(dep_id)
                queue.extend(dep_task.dependents)
        
        return to_skip

    def get_execution_order(self) -> list[list[str]]:
        """Get tasks grouped by execution level (for parallel execution).
        
        Returns:
            List of task ID lists, where each inner list can run in parallel.
        """
        levels: list[list[str]] = []
        remaining = set(self.nodes.keys())
        completed: set[str] = set()
        
        while remaining:
            # Find tasks with all dependencies in completed set
            current_level: list[str] = []
            for task_id in remaining:
                task = self.nodes[task_id]
                if all(dep_id in completed for dep_id in task.dependencies):
                    current_level.append(task_id)
            
            if not current_level:
                # Cycle detected or error
                break
            
            # Sort by priority
            current_level.sort(
                key=lambda tid: self.nodes[tid].priority, reverse=True
            )
            levels.append(current_level)
            
            # Move to completed
            for task_id in current_level:
                completed.add(task_id)
                remaining.discard(task_id)
        
        return levels

    def get_critical_path(self) -> list[str]:
        """Calculate the critical path (longest duration path).
        
        Returns:
            List of task IDs on the critical path.
        """
        if not self.nodes:
            return []
        
        # Calculate earliest completion time for each task
        earliest: dict[str, float] = {}
        
        # Process in topological order
        for level in self.get_execution_order():
            for task_id in level:
                task = self.nodes[task_id]
                # Earliest start is max of all dependency completions
                earliest_start = 0.0
                for dep_id in task.dependencies:
                    if dep_id in earliest:
                        earliest_start = max(earliest_start, earliest[dep_id])
                earliest[task_id] = earliest_start + task.estimated_duration_ms
        
        # Find the task with latest completion
        if not earliest:
            return []
        
        end_task = max(earliest.keys(), key=lambda k: earliest[k])
        
        # Trace back the critical path
        path: list[str] = [end_task]
        current = end_task
        
        while True:
            task = self.nodes[current]
            if not task.dependencies:
                break
            
            # Find the dependency on the critical path
            max_time = 0.0
            critical_dep = None
            for dep_id in task.dependencies:
                if dep_id in earliest and earliest[dep_id] > max_time:
                    max_time = earliest[dep_id]
                    critical_dep = dep_id
            
            if critical_dep:
                path.insert(0, critical_dep)
                current = critical_dep
            else:
                break
        
        return path

    def estimate_total_duration(self, max_parallel: int = 1) -> float:
        """Estimate total execution duration with parallelism.
        
        Args:
            max_parallel: Maximum parallel task count.
            
        Returns:
            Estimated duration in milliseconds.
        """
        total_duration = 0.0
        
        for level in self.get_execution_order():
            # Split level into batches based on max_parallel
            level_tasks = [self.nodes[tid] for tid in level]
            for i in range(0, len(level_tasks), max_parallel):
                batch = level_tasks[i:i + max_parallel]
                # Batch duration is max of all tasks in batch
                batch_duration = max(
                    t.estimated_duration_ms for t in batch
                ) if batch else 0
                total_duration += batch_duration
        
        return total_duration

    def to_dict(self) -> dict[str, Any]:
        """Convert DAG to dictionary representation."""
        return {
            "nodes": {tid: t.to_dict() for tid, t in self.nodes.items()},
            "root_tasks": self.root_tasks,
            "execution_order": self.get_execution_order(),
            "critical_path": self.get_critical_path(),
            "estimated_duration_ms": self.estimate_total_duration(max_parallel=4),
        }


class Planner:
    """LLM-assisted planner that produces a DAG execution plan from an objective."""

    def __init__(
        self, state_machine: ExecutionStateMachine, plan_fn: PlanFunction | None = None
    ):
        self.state_machine = state_machine
        self.plan_fn = plan_fn
        self.logger = get_logger("planner")

    def plan(self, objective: str) -> dict[str, Any]:
        """Generate a plan for the given objective using the configured model."""

        self.state_machine.start()
        try:
            if not self.plan_fn:
                # Generate a meaningful plan when no model is provided
                plan = self._generate_default_plan(objective)
            else:
                plan = self.plan_fn(objective)

            self.state_machine.plan_complete()
            self.logger.info("planning.complete", plan_summary=list(plan.keys()))
            return plan
        except Exception as exc:  # noqa: BLE001
            self.state_machine.plan_failed()
            self.logger.error("planning.failed", error=str(exc))
            raise

    def plan_as_dag(self, objective: str) -> ExecutionDAG:
        """Generate a DAG-based execution plan for optimal parallelization.
        
        Args:
            objective: The high-level objective to plan for.
            
        Returns:
            ExecutionDAG with all tasks and dependencies.
        """
        self.state_machine.start()
        try:
            # Get the basic plan
            if self.plan_fn:
                basic_plan = self.plan_fn(objective)
            else:
                basic_plan = self._generate_default_plan(objective)
            
            # Convert to DAG
            dag = self._convert_plan_to_dag(basic_plan)
            
            # Validate DAG
            is_valid, errors = dag.validate()
            if not is_valid:
                self.logger.warning("dag.validation_warnings", errors=errors)
            
            self.state_machine.plan_complete()
            self.logger.info(
                "dag_planning.complete",
                tasks=len(dag.nodes),
                levels=len(dag.get_execution_order()),
                critical_path_length=len(dag.get_critical_path()),
            )
            return dag
            
        except Exception as exc:
            self.state_machine.plan_failed()
            self.logger.error("dag_planning.failed", error=str(exc))
            raise

    def _convert_plan_to_dag(self, plan: dict[str, Any]) -> ExecutionDAG:
        """Convert a basic plan to an ExecutionDAG.
        
        Args:
            plan: Basic plan dictionary with steps.
            
        Returns:
            ExecutionDAG with proper dependencies.
        """
        dag = ExecutionDAG()
        steps = plan.get("steps", [])
        backends = plan.get("backends", [])
        execution_mode = plan.get("execution_mode", "single")
        
        task_id_map: dict[int, str] = {}
        
        for step in steps:
            step_num = step.get("step", 0)
            action = step.get("action", "")
            
            if action == "execute" and execution_mode == "comparison" and backends:
                # Create parallel execution tasks for each backend
                execute_task_ids: list[str] = []
                
                for backend in backends:
                    task = TaskNode(
                        action="execute",
                        description=f"Execute on {backend}",
                        parameters={
                            **step.get("parameters", {}),
                            "backend": backend,
                        },
                        dependencies=self._get_dependencies(step_num, task_id_map),
                        priority=100 - step_num,
                        estimated_duration_ms=500.0,  # Backend execution typically slower
                        tags=["execution", backend],
                    )
                    dag.add_task(task)
                    execute_task_ids.append(task.task_id)
                
                # Store all execution tasks for dependency resolution
                task_id_map[step_num] = execute_task_ids[0]  # Primary reference
                
                # Create aggregation point for collect_results
                for tid in execute_task_ids[1:]:
                    # Mark as same step for dependency resolution
                    pass
                
            else:
                # Single task
                task = TaskNode(
                    action=action,
                    description=step.get("description", action),
                    parameters=step.get("parameters", {}),
                    dependencies=self._get_dependencies(step_num, task_id_map),
                    priority=100 - step_num,
                    estimated_duration_ms=self._estimate_task_duration(action),
                    tags=self._get_task_tags(action),
                )
                dag.add_task(task)
                task_id_map[step_num] = task.task_id
        
        return dag

    def _get_dependencies(
        self, step_num: int, task_id_map: dict[int, str]
    ) -> list[str]:
        """Get dependencies for a step based on step number."""
        if step_num <= 1:
            return []
        
        # Depend on previous step
        prev_step = step_num - 1
        if prev_step in task_id_map:
            return [task_id_map[prev_step]]
        
        return []

    def _estimate_task_duration(self, action: str) -> float:
        """Estimate task duration based on action type."""
        durations = {
            "create_circuit": 10.0,
            "execute": 500.0,
            "collect_results": 50.0,
            "compare": 100.0,
            "analyze": 200.0,
            "export": 100.0,
        }
        return durations.get(action, 100.0)

    def _get_task_tags(self, action: str) -> list[str]:
        """Get tags for a task based on action type."""
        tag_map = {
            "create_circuit": ["circuit", "initialization"],
            "execute": ["execution", "backend"],
            "collect_results": ["results", "aggregation"],
            "compare": ["analysis", "comparison"],
            "analyze": ["analysis", "insights"],
            "export": ["output", "export"],
        }
        return tag_map.get(action, [action])

    def _generate_default_plan(self, objective: str) -> dict[str, Any]:
        """Generate a default plan based on objective keywords.

        Analyzes the objective to determine:
        - Circuit type (bell, ghz, teleportation, etc.)
        - Execution mode (single, comparison)
        - Required backends
        - Number of shots
        """
        objective_lower = objective.lower()

        # Determine circuit type from objective
        circuit_type = "bell"  # default
        qubits = 2
        if "ghz" in objective_lower:
            circuit_type = "ghz"
            qubits = 3
            # Try to extract qubit count
            import re

            match = re.search(r"(\d+)[-\s]*qubit", objective_lower)
            if match:
                qubits = int(match.group(1))
        elif "teleport" in objective_lower:
            circuit_type = "teleportation"
            qubits = 3
        elif "superposition" in objective_lower or "hadamard" in objective_lower:
            circuit_type = "superposition"
            qubits = 1
        elif "entangle" in objective_lower:
            circuit_type = "bell"
            qubits = 2

        # Determine execution mode
        execution_mode = "single"
        backends = []
        if "compare" in objective_lower or "comparison" in objective_lower:
            execution_mode = "comparison"
            backends = ["cirq", "qiskit"]
        elif "all backend" in objective_lower:
            execution_mode = "comparison"
            backends = ["cirq", "qiskit", "lret"]

        # Extract shots if mentioned
        shots = 1024
        import re

        shots_match = re.search(r"(\d+)\s*shots?", objective_lower)
        if shots_match:
            shots = int(shots_match.group(1))

        # Build plan steps
        steps = [
            {
                "step": 1,
                "action": "create_circuit",
                "description": f"Create {circuit_type} circuit with {qubits} qubits",
                "parameters": {"circuit_type": circuit_type, "qubits": qubits},
            },
            {
                "step": 2,
                "action": "execute",
                "description": f"Execute circuit with {shots} shots",
                "parameters": {"shots": shots, "backends": backends or ["auto"]},
            },
            {
                "step": 3,
                "action": "collect_results",
                "description": "Collect and normalize results",
                "parameters": {},
            },
        ]

        if execution_mode == "comparison":
            steps.append(
                {
                    "step": 4,
                    "action": "compare",
                    "description": "Compare results across backends",
                    "parameters": {"backends": backends},
                }
            )

        return {
            "objective": objective,
            "circuit_type": circuit_type,
            "qubits": qubits,
            "shots": shots,
            "execution_mode": execution_mode,
            "backends": backends,
            "steps": steps,
            "generated_by": "default_planner",
        }


__all__ = [
    "TaskStatus",
    "TaskNode",
    "ExecutionDAG",
    "Planner",
    "PlanFunction",
]
