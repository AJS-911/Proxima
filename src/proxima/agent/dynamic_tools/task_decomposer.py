"""Task Decomposer for Multi-Step Operations.

This module implements Phase 3.1.1: Task Decomposition for the Dynamic AI Assistant.
It provides hierarchical task decomposition using LLM reasoning to break down
complex user requests into executable steps.

Key Features:
============
- Hierarchical task decomposition using tree structures
- Dependency graph construction for operation ordering
- Critical path analysis for optimization
- Parallel execution detection
- Resource requirement estimation
- Time estimation for operation sequences

Design Principle:
================
All decomposition uses LLM reasoning - NO hardcoded patterns or keyword matching.
The LLM analyzes the request and determines the optimal breakdown dynamically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
import uuid

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks in the decomposition tree."""
    COMPOSITE = "composite"  # Task with subtasks
    ATOMIC = "atomic"  # Single tool execution
    CONDITIONAL = "conditional"  # Branch based on condition
    LOOP = "loop"  # Repeated execution
    PARALLEL = "parallel"  # Parallel subtasks


class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class ResourceType(Enum):
    """Types of resources tasks may need."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK = "network"
    GPU = "gpu"
    TERMINAL = "terminal"
    FILE_HANDLE = "file_handle"


@dataclass
class ResourceRequirement:
    """Resource requirement specification."""
    resource_type: ResourceType
    amount: float  # 0.0-1.0 for percentage, or absolute value
    is_percentage: bool = True
    blocking: bool = False  # If True, resource is exclusively held


@dataclass
class TaskNode:
    """A node in the task decomposition tree.
    
    Represents either a composite task (with children) or an atomic task
    (a single tool execution).
    """
    task_id: str
    name: str
    description: str
    task_type: TaskType = TaskType.ATOMIC
    
    # For atomic tasks
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # For composite tasks
    children: List["TaskNode"] = field(default_factory=list)
    
    # For conditional tasks
    condition: Optional[str] = None  # LLM-evaluated condition
    true_branch: Optional["TaskNode"] = None
    false_branch: Optional["TaskNode"] = None
    
    # For loop tasks
    loop_condition: Optional[str] = None
    max_iterations: int = 10
    
    # Dependencies (task IDs that must complete first)
    dependencies: List[str] = field(default_factory=list)
    
    # Metadata
    priority: TaskPriority = TaskPriority.NORMAL
    estimated_duration_ms: Optional[int] = None
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    
    # Execution state
    is_executed: bool = False
    execution_result: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.task_id:
            self.task_id = f"task_{uuid.uuid4().hex[:8]}"
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def get_all_descendants(self) -> List["TaskNode"]:
        """Get all descendant nodes."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type.value,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "children": [c.to_dict() for c in self.children],
            "dependencies": self.dependencies,
            "priority": self.priority.value,
            "estimated_duration_ms": self.estimated_duration_ms,
            "is_executed": self.is_executed,
            "error": self.error,
        }


@dataclass
class DependencyEdge:
    """An edge in the dependency graph."""
    from_task: str
    to_task: str
    edge_type: str = "depends_on"  # depends_on, data_flow, resource
    weight: float = 1.0  # For critical path calculation


@dataclass
class DependencyGraph:
    """Graph representing task dependencies."""
    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    edges: List[DependencyEdge] = field(default_factory=list)
    
    def add_node(self, node: TaskNode):
        """Add a node to the graph."""
        self.nodes[node.task_id] = node
    
    def add_edge(self, from_id: str, to_id: str, edge_type: str = "depends_on"):
        """Add an edge between nodes."""
        self.edges.append(DependencyEdge(
            from_task=from_id,
            to_task=to_id,
            edge_type=edge_type,
        ))
    
    def get_roots(self) -> List[TaskNode]:
        """Get nodes with no dependencies (can start immediately)."""
        dependent_ids = {e.to_task for e in self.edges}
        return [n for n in self.nodes.values() if n.task_id not in dependent_ids]
    
    def get_dependencies(self, task_id: str) -> List[str]:
        """Get all task IDs that must complete before this task."""
        return [e.from_task for e in self.edges if e.to_task == task_id]
    
    def get_dependents(self, task_id: str) -> List[str]:
        """Get all task IDs that depend on this task."""
        return [e.to_task for e in self.edges if e.from_task == task_id]
    
    def topological_sort(self) -> List[str]:
        """Get tasks in execution order (topological sort)."""
        in_degree = {n: 0 for n in self.nodes}
        for edge in self.edges:
            if edge.to_task in in_degree:
                in_degree[edge.to_task] += 1
        
        # Start with nodes that have no dependencies
        queue = [n for n, d in in_degree.items() if d == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for edge in self.edges:
                if edge.from_task == current:
                    in_degree[edge.to_task] -= 1
                    if in_degree[edge.to_task] == 0:
                        queue.append(edge.to_task)
        
        return result
    
    def find_parallel_groups(self) -> List[List[str]]:
        """Find groups of tasks that can execute in parallel."""
        groups = []
        completed: Set[str] = set()
        all_tasks = set(self.nodes.keys())
        
        while completed != all_tasks:
            # Find all tasks whose dependencies are satisfied
            ready = []
            for task_id in all_tasks - completed:
                deps = self.get_dependencies(task_id)
                if all(d in completed for d in deps):
                    ready.append(task_id)
            
            if ready:
                groups.append(ready)
                completed.update(ready)
            else:
                # Cycle detected or error
                break
        
        return groups
    
    def calculate_critical_path(self) -> Tuple[List[str], int]:
        """Calculate the critical path (longest execution path).
        
        Returns:
            Tuple of (path as list of task IDs, total duration in ms)
        """
        # Calculate earliest start times
        earliest_start: Dict[str, int] = {}
        sorted_tasks = self.topological_sort()
        
        for task_id in sorted_tasks:
            node = self.nodes[task_id]
            deps = self.get_dependencies(task_id)
            
            if not deps:
                earliest_start[task_id] = 0
            else:
                max_dep_end = 0
                for dep in deps:
                    dep_node = self.nodes[dep]
                    dep_duration = dep_node.estimated_duration_ms or 1000
                    dep_end = earliest_start.get(dep, 0) + dep_duration
                    max_dep_end = max(max_dep_end, dep_end)
                earliest_start[task_id] = max_dep_end
        
        # Find the path with maximum duration
        # Start from tasks with no dependents and trace back
        end_tasks = [t for t in self.nodes if not self.get_dependents(t)]
        
        max_duration = 0
        critical_path: List[str] = []
        
        for end_task in end_tasks:
            end_node = self.nodes[end_task]
            end_duration = end_node.estimated_duration_ms or 1000
            total = earliest_start[end_task] + end_duration
            
            if total > max_duration:
                max_duration = total
                # Reconstruct path
                path = [end_task]
                current = end_task
                
                while True:
                    deps = self.get_dependencies(current)
                    if not deps:
                        break
                    
                    # Find the dependency that ends latest
                    latest_dep = None
                    latest_end = -1
                    
                    for dep in deps:
                        dep_node = self.nodes[dep]
                        dep_duration = dep_node.estimated_duration_ms or 1000
                        dep_end = earliest_start[dep] + dep_duration
                        
                        if dep_end > latest_end:
                            latest_end = dep_end
                            latest_dep = dep
                    
                    if latest_dep:
                        path.insert(0, latest_dep)
                        current = latest_dep
                    else:
                        break
                
                critical_path = path
        
        return critical_path, max_duration


@dataclass 
class DecompositionResult:
    """Result of task decomposition."""
    root_task: TaskNode
    dependency_graph: DependencyGraph
    
    # Analysis results
    total_tasks: int = 0
    atomic_tasks: int = 0
    parallel_groups: List[List[str]] = field(default_factory=list)
    critical_path: List[str] = field(default_factory=list)
    estimated_total_duration_ms: int = 0
    
    # Metadata
    decomposition_reasoning: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "root_task": self.root_task.to_dict(),
            "total_tasks": self.total_tasks,
            "atomic_tasks": self.atomic_tasks,
            "parallel_groups": self.parallel_groups,
            "critical_path": self.critical_path,
            "estimated_total_duration_ms": self.estimated_total_duration_ms,
            "decomposition_reasoning": self.decomposition_reasoning,
            "created_at": self.created_at,
        }


class TaskDecomposer:
    """Decomposes complex requests into executable task trees.
    
    Uses LLM reasoning to:
    1. Understand the user's request
    2. Break it down into logical steps
    3. Identify dependencies between steps
    4. Detect opportunities for parallelism
    5. Estimate resource requirements and duration
    
    Example:
        >>> decomposer = TaskDecomposer(llm_client=client)
        >>> result = await decomposer.decompose(
        ...     "Clone the repository, build it, and run the tests"
        ... )
        >>> print(result.parallel_groups)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
    ):
        """Initialize the decomposer.
        
        Args:
            llm_client: LLM client for reasoning
            tool_registry: Tool registry for available tools
        """
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        
        # Get registry if not provided
        if self._tool_registry is None:
            from .tool_registry import get_tool_registry
            self._tool_registry = get_tool_registry()
    
    async def decompose(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        max_depth: int = 5,
    ) -> DecompositionResult:
        """Decompose a complex request into tasks.
        
        Args:
            request: The user's natural language request
            context: Optional context (workspace state, etc.)
            max_depth: Maximum decomposition depth
            
        Returns:
            DecompositionResult with task tree and analysis
        """
        # Get available tools for LLM context
        available_tools = self._get_tool_descriptions()
        
        # Use LLM to decompose the request
        decomposition = await self._llm_decompose(
            request,
            available_tools,
            context or {},
            max_depth,
        )
        
        # Build dependency graph
        graph = self._build_dependency_graph(decomposition["root_task"])
        
        # Analyze the graph
        parallel_groups = graph.find_parallel_groups()
        critical_path, total_duration = graph.calculate_critical_path()
        
        # Count tasks
        all_tasks = [decomposition["root_task"]]
        all_tasks.extend(decomposition["root_task"].get_all_descendants())
        atomic_count = sum(1 for t in all_tasks if t.task_type == TaskType.ATOMIC)
        
        return DecompositionResult(
            root_task=decomposition["root_task"],
            dependency_graph=graph,
            total_tasks=len(all_tasks),
            atomic_tasks=atomic_count,
            parallel_groups=parallel_groups,
            critical_path=critical_path,
            estimated_total_duration_ms=total_duration,
            decomposition_reasoning=decomposition.get("reasoning", ""),
        )
    
    def _get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of available tools."""
        tools = []
        
        for registered in self._tool_registry.get_all_tools():
            defn = registered.definition
            
            # Handle category as string or enum
            category = defn.category
            if hasattr(category, 'value'):
                category = category.value
            
            tools.append({
                "name": defn.name,
                "description": defn.description,
                "category": category,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.param_type.value if hasattr(p.param_type, 'value') else str(p.param_type),
                        "required": p.required,
                        "description": p.description,
                    }
                    for p in defn.parameters
                ],
            })
        
        return tools
    
    async def _llm_decompose(
        self,
        request: str,
        available_tools: List[Dict[str, Any]],
        context: Dict[str, Any],
        max_depth: int,
    ) -> Dict[str, Any]:
        """Use LLM to decompose the request.
        
        Returns a dictionary with 'root_task' and 'reasoning'.
        """
        if self._llm_client is None:
            # Fallback: Simple single-task decomposition
            return self._simple_decompose(request)
        
        # Build prompt for LLM
        tools_text = json.dumps(available_tools, indent=2)
        context_text = json.dumps(context, indent=2) if context else "{}"
        
        prompt = f"""You are a task decomposition system. Break down the user's request into a tree of executable tasks.

User Request: {request}

Available Tools:
{tools_text}

Current Context:
{context_text}

Instructions:
1. Analyze the request and identify all required operations
2. Break complex operations into simpler subtasks
3. Identify dependencies between tasks (what must complete before another can start)
4. Identify tasks that can run in parallel (no dependencies between them)
5. Estimate duration for each task in milliseconds
6. Map each atomic task to an available tool

Respond with a JSON object in this exact format:
{{
    "reasoning": "Your step-by-step reasoning about the decomposition",
    "root_task": {{
        "name": "Main Task Name",
        "description": "What this task accomplishes",
        "task_type": "composite|atomic|parallel",
        "children": [
            {{
                "name": "Subtask 1",
                "description": "What subtask 1 does",
                "task_type": "atomic",
                "tool_name": "name_of_tool_to_use",
                "parameters": {{"param1": "value1"}},
                "dependencies": [],
                "estimated_duration_ms": 1000
            }},
            {{
                "name": "Subtask 2", 
                "description": "What subtask 2 does",
                "task_type": "atomic",
                "tool_name": "another_tool",
                "parameters": {{}},
                "dependencies": ["task_id_of_subtask_1"],
                "estimated_duration_ms": 2000
            }}
        ]
    }}
}}

Important:
- Use actual tool names from the available tools list
- Set dependencies correctly (task must wait for dependency to complete)
- If tasks can run in parallel (no dependencies), mark task_type as "parallel" for the parent
- Each atomic task must have a tool_name
- Maximum decomposition depth: {max_depth}
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            
            # Parse JSON from response
            from .structured_output import JSONExtractor
            extractor = JSONExtractor()
            parsed, _ = extractor.extract(response)
            
            if parsed and "root_task" in parsed:
                # Convert to TaskNode tree
                root = self._dict_to_task_node(parsed["root_task"])
                return {
                    "root_task": root,
                    "reasoning": parsed.get("reasoning", ""),
                }
                
        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}")
        
        # Fallback
        return self._simple_decompose(request)
    
    def _dict_to_task_node(
        self,
        data: Dict[str, Any],
        parent_id: Optional[str] = None,
    ) -> TaskNode:
        """Convert dictionary to TaskNode."""
        task_id = data.get("task_id", f"task_{uuid.uuid4().hex[:8]}")
        
        task_type_str = data.get("task_type", "atomic")
        task_type = TaskType.ATOMIC
        if task_type_str == "composite":
            task_type = TaskType.COMPOSITE
        elif task_type_str == "parallel":
            task_type = TaskType.PARALLEL
        elif task_type_str == "conditional":
            task_type = TaskType.CONDITIONAL
        elif task_type_str == "loop":
            task_type = TaskType.LOOP
        
        node = TaskNode(
            task_id=task_id,
            name=data.get("name", "Unnamed Task"),
            description=data.get("description", ""),
            task_type=task_type,
            tool_name=data.get("tool_name"),
            parameters=data.get("parameters", {}),
            dependencies=data.get("dependencies", []),
            estimated_duration_ms=data.get("estimated_duration_ms"),
        )
        
        # Process children
        for child_data in data.get("children", []):
            child = self._dict_to_task_node(child_data, task_id)
            node.children.append(child)
        
        return node
    
    def _simple_decompose(self, request: str) -> Dict[str, Any]:
        """Simple fallback decomposition without LLM."""
        # Create a single task that attempts to find the best tool
        root = TaskNode(
            task_id="root",
            name="Execute Request",
            description=request,
            task_type=TaskType.ATOMIC,
            estimated_duration_ms=5000,
        )
        
        return {
            "root_task": root,
            "reasoning": "Simple single-task decomposition (LLM not available)",
        }
    
    def _build_dependency_graph(self, root: TaskNode) -> DependencyGraph:
        """Build dependency graph from task tree."""
        graph = DependencyGraph()
        
        # Collect all tasks
        all_tasks = [root]
        all_tasks.extend(root.get_all_descendants())
        
        # Add all nodes
        for task in all_tasks:
            graph.add_node(task)
        
        # Add edges for explicit dependencies
        for task in all_tasks:
            for dep_id in task.dependencies:
                graph.add_edge(dep_id, task.task_id)
        
        # Add implicit edges from parent-child relationships
        # Children of composite nodes depend on each other sequentially
        # unless parent is PARALLEL type
        def add_child_edges(parent: TaskNode):
            if parent.task_type == TaskType.PARALLEL:
                # Parallel children have no inter-dependencies
                pass
            elif parent.task_type in (TaskType.COMPOSITE, TaskType.LOOP):
                # Sequential children depend on previous
                for i in range(1, len(parent.children)):
                    prev = parent.children[i - 1]
                    curr = parent.children[i]
                    if prev.task_id not in curr.dependencies:
                        graph.add_edge(prev.task_id, curr.task_id)
            
            # Recurse
            for child in parent.children:
                add_child_edges(child)
        
        add_child_edges(root)
        
        return graph
    
    def estimate_resources(
        self,
        result: DecompositionResult,
    ) -> Dict[str, Any]:
        """Estimate resource requirements for the decomposition.
        
        Returns dictionary with:
        - total_cpu: Estimated CPU usage
        - total_memory_mb: Estimated memory in MB
        - disk_operations: Number of disk operations
        - network_operations: Number of network operations
        """
        total_cpu = 0.0
        total_memory = 0
        disk_ops = 0
        network_ops = 0
        
        all_tasks = [result.root_task]
        all_tasks.extend(result.root_task.get_all_descendants())
        
        for task in all_tasks:
            if task.task_type != TaskType.ATOMIC:
                continue
            
            # Estimate based on tool category
            tool = self._tool_registry.get_tool(task.tool_name) if task.tool_name else None
            
            if tool:
                category = tool.definition.category
                if hasattr(category, 'value'):
                    category = category.value
                
                if category == "file_system":
                    disk_ops += 1
                    total_cpu += 0.1
                    total_memory += 10
                elif category == "git":
                    disk_ops += 2
                    network_ops += 1
                    total_cpu += 0.2
                    total_memory += 50
                elif category == "terminal":
                    total_cpu += 0.5
                    total_memory += 100
                else:
                    total_cpu += 0.1
                    total_memory += 20
        
        return {
            "total_cpu": min(total_cpu, 1.0),
            "total_memory_mb": total_memory,
            "disk_operations": disk_ops,
            "network_operations": network_ops,
        }


# Module-level decomposer instance
_global_decomposer: Optional[TaskDecomposer] = None


def get_task_decomposer(
    llm_client: Optional[Any] = None,
) -> TaskDecomposer:
    """Get the global task decomposer.
    
    Args:
        llm_client: Optional LLM client
        
    Returns:
        TaskDecomposer instance
    """
    global _global_decomposer
    if _global_decomposer is None:
        _global_decomposer = TaskDecomposer(llm_client=llm_client)
    return _global_decomposer
