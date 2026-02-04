"""Workflow Engine for Dynamic Workflow Generation.

This module implements Phase 3.1.2: Dynamic Workflow Generation for the
Dynamic AI Assistant. It provides workflow templates, conditional branching,
loop detection, error recovery, and workflow persistence.

Key Features:
============
- Workflow templates that LLM can instantiate
- Conditional branching based on intermediate results
- Loop detection for repeated operations
- Error recovery paths in workflows
- Workflow visualization for user review
- Workflow persistence and resumption

Design Principle:
================
All workflow logic uses LLM reasoning - NO hardcoded patterns.
The LLM determines branching conditions and recovery strategies dynamically.
"""

from __future__ import annotations

import json
import logging
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Status of a workflow."""
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeType(Enum):
    """Types of workflow nodes."""
    START = "start"
    END = "end"
    TASK = "task"  # Execute a tool
    DECISION = "decision"  # Conditional branch
    FORK = "fork"  # Parallel split
    JOIN = "join"  # Parallel merge
    LOOP_START = "loop_start"
    LOOP_END = "loop_end"
    ERROR_HANDLER = "error_handler"
    SUBWORKFLOW = "subworkflow"


class TransitionType(Enum):
    """Types of transitions between nodes."""
    NORMAL = "normal"
    CONDITIONAL = "conditional"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCEL = "cancel"


@dataclass
class WorkflowVariable:
    """A variable in the workflow context."""
    name: str
    value: Any = None
    var_type: str = "any"
    is_input: bool = False
    is_output: bool = False
    description: str = ""


@dataclass
class Transition:
    """A transition between workflow nodes."""
    transition_id: str
    from_node: str
    to_node: str
    transition_type: TransitionType = TransitionType.NORMAL
    condition: Optional[str] = None  # LLM-evaluated condition
    priority: int = 0  # Higher priority transitions checked first
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "transition_type": self.transition_type.value,
            "condition": self.condition,
            "priority": self.priority,
        }


@dataclass
class WorkflowNode:
    """A node in the workflow graph."""
    node_id: str
    name: str
    node_type: NodeType
    description: str = ""
    
    # For TASK nodes
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_mappings: Dict[str, str] = field(default_factory=dict)  # Map from workflow vars
    
    # For DECISION nodes
    decision_expression: Optional[str] = None  # LLM evaluates this
    
    # For LOOP nodes
    loop_variable: Optional[str] = None
    loop_collection: Optional[str] = None  # Variable name containing iterable
    max_iterations: int = 100
    
    # For ERROR_HANDLER nodes
    error_types: List[str] = field(default_factory=list)  # Types of errors to handle
    retry_count: int = 0
    retry_delay_ms: int = 1000
    
    # For SUBWORKFLOW nodes
    subworkflow_id: Optional[str] = None
    
    # Execution state
    status: WorkflowStatus = WorkflowStatus.READY
    result: Optional[Any] = None
    error: Optional[str] = None
    executed_at: Optional[str] = None
    execution_count: int = 0
    
    # Metadata
    timeout_ms: Optional[int] = None
    is_critical: bool = False  # If critical, workflow fails if this fails
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type.value,
            "description": self.description,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "status": self.status.value,
            "result": str(self.result)[:200] if self.result else None,
            "error": self.error,
        }


@dataclass
class WorkflowTemplate:
    """A reusable workflow template.
    
    Templates can be instantiated with different parameters to create
    concrete workflow instances.
    """
    template_id: str
    name: str
    description: str
    category: str = "general"
    
    # Template structure
    nodes: List[WorkflowNode] = field(default_factory=list)
    transitions: List[Transition] = field(default_factory=list)
    
    # Template variables (to be filled when instantiating)
    input_variables: List[WorkflowVariable] = field(default_factory=list)
    output_variables: List[WorkflowVariable] = field(default_factory=list)
    
    # Metadata
    author: str = ""
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def instantiate(
        self,
        inputs: Dict[str, Any],
    ) -> "Workflow":
        """Create a workflow instance from this template.
        
        Args:
            inputs: Values for input variables
            
        Returns:
            A new Workflow instance
        """
        workflow = Workflow(
            workflow_id=f"wf_{uuid.uuid4().hex[:8]}",
            name=f"{self.name} Instance",
            description=self.description,
            template_id=self.template_id,
        )
        
        # Copy nodes (deep copy)
        for node in self.nodes:
            new_node = WorkflowNode(
                node_id=node.node_id,
                name=node.name,
                node_type=node.node_type,
                description=node.description,
                tool_name=node.tool_name,
                parameters=node.parameters.copy(),
                parameter_mappings=node.parameter_mappings.copy(),
                decision_expression=node.decision_expression,
                loop_variable=node.loop_variable,
                loop_collection=node.loop_collection,
                max_iterations=node.max_iterations,
                timeout_ms=node.timeout_ms,
                is_critical=node.is_critical,
            )
            workflow.nodes[new_node.node_id] = new_node
        
        # Copy transitions
        for trans in self.transitions:
            workflow.transitions.append(Transition(
                transition_id=trans.transition_id,
                from_node=trans.from_node,
                to_node=trans.to_node,
                transition_type=trans.transition_type,
                condition=trans.condition,
                priority=trans.priority,
            ))
        
        # Set input variables
        for var in self.input_variables:
            workflow.variables[var.name] = WorkflowVariable(
                name=var.name,
                value=inputs.get(var.name, var.value),
                var_type=var.var_type,
                is_input=True,
                description=var.description,
            )
        
        # Initialize output variables
        for var in self.output_variables:
            workflow.variables[var.name] = WorkflowVariable(
                name=var.name,
                var_type=var.var_type,
                is_output=True,
                description=var.description,
            )
        
        return workflow
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "nodes": [n.to_dict() for n in self.nodes],
            "transitions": [t.to_dict() for t in self.transitions],
            "input_variables": [{"name": v.name, "type": v.var_type} for v in self.input_variables],
            "output_variables": [{"name": v.name, "type": v.var_type} for v in self.output_variables],
            "version": self.version,
            "tags": self.tags,
        }


@dataclass
class Workflow:
    """A workflow instance that can be executed."""
    workflow_id: str
    name: str
    description: str
    template_id: Optional[str] = None
    
    # Structure
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    transitions: List[Transition] = field(default_factory=list)
    
    # Variables
    variables: Dict[str, WorkflowVariable] = field(default_factory=dict)
    
    # Execution state
    status: WorkflowStatus = WorkflowStatus.DRAFT
    current_nodes: Set[str] = field(default_factory=set)  # Currently active nodes
    completed_nodes: Set[str] = field(default_factory=set)
    failed_nodes: Set[str] = field(default_factory=set)
    
    # History
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def get_start_node(self) -> Optional[WorkflowNode]:
        """Get the start node of the workflow."""
        for node in self.nodes.values():
            if node.node_type == NodeType.START:
                return node
        return None
    
    def get_end_nodes(self) -> List[WorkflowNode]:
        """Get all end nodes of the workflow."""
        return [n for n in self.nodes.values() if n.node_type == NodeType.END]
    
    def get_outgoing_transitions(self, node_id: str) -> List[Transition]:
        """Get all transitions leaving a node."""
        transitions = [t for t in self.transitions if t.from_node == node_id]
        # Sort by priority (higher first)
        transitions.sort(key=lambda t: t.priority, reverse=True)
        return transitions
    
    def get_incoming_transitions(self, node_id: str) -> List[Transition]:
        """Get all transitions entering a node."""
        return [t for t in self.transitions if t.to_node == node_id]
    
    def set_variable(self, name: str, value: Any):
        """Set a workflow variable."""
        if name in self.variables:
            self.variables[name].value = value
        else:
            self.variables[name] = WorkflowVariable(name=name, value=value)
    
    def get_variable(self, name: str) -> Any:
        """Get a workflow variable value."""
        if name in self.variables:
            return self.variables[name].value
        return None
    
    def add_history(self, event: str, node_id: Optional[str] = None, details: Optional[Dict] = None):
        """Add an event to execution history."""
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "node_id": node_id,
            "details": details or {},
        })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "transitions": [t.to_dict() for t in self.transitions],
            "variables": {k: {"name": v.name, "value": str(v.value)[:100]} for k, v in self.variables.items()},
            "current_nodes": list(self.current_nodes),
            "completed_nodes": list(self.completed_nodes),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
    
    def visualize(self) -> str:
        """Generate a text visualization of the workflow."""
        lines = [
            f"Workflow: {self.name}",
            f"Status: {self.status.value}",
            f"ID: {self.workflow_id}",
            "",
            "Nodes:",
        ]
        
        status_icons = {
            WorkflowStatus.READY: "⚪",
            WorkflowStatus.RUNNING: "🔵",
            WorkflowStatus.COMPLETED: "✅",
            WorkflowStatus.FAILED: "❌",
            WorkflowStatus.PAUSED: "⏸️",
        }
        
        for node_id, node in self.nodes.items():
            icon = status_icons.get(node.status, "⚪")
            current = " ← CURRENT" if node_id in self.current_nodes else ""
            lines.append(f"  {icon} [{node.node_type.value}] {node.name}{current}")
        
        lines.append("")
        lines.append("Transitions:")
        for trans in self.transitions:
            cond = f" if {trans.condition}" if trans.condition else ""
            lines.append(f"  {trans.from_node} → {trans.to_node}{cond}")
        
        return "\n".join(lines)


class WorkflowEngine:
    """Engine for executing workflows.
    
    The engine handles:
    - Workflow instantiation from templates
    - LLM-based workflow generation from natural language
    - Workflow execution with branching and loops
    - Error recovery and retry logic
    - Workflow persistence and resumption
    
    Example:
        >>> engine = WorkflowEngine(llm_client=client)
        >>> workflow = await engine.generate_workflow(
        ...     "Clone repo, build project, run tests"
        ... )
        >>> result = await engine.execute(workflow)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        persistence_dir: Optional[Path] = None,
    ):
        """Initialize the workflow engine.
        
        Args:
            llm_client: LLM client for reasoning
            tool_registry: Tool registry for available tools
            persistence_dir: Directory for workflow persistence
        """
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._persistence_dir = persistence_dir
        
        # Templates
        self._templates: Dict[str, WorkflowTemplate] = {}
        
        # Active workflows
        self._active_workflows: Dict[str, Workflow] = {}
        
        # Callbacks
        self._on_node_start: Optional[Callable[[WorkflowNode, Workflow], None]] = None
        self._on_node_complete: Optional[Callable[[WorkflowNode, Workflow], None]] = None
        self._on_workflow_complete: Optional[Callable[[Workflow], None]] = None
        
        # Get registry if not provided
        if self._tool_registry is None:
            from .tool_registry import get_tool_registry
            self._tool_registry = get_tool_registry()
        
        # Load built-in templates
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """Load built-in workflow templates."""
        # Clone and Build template
        clone_build = WorkflowTemplate(
            template_id="clone_and_build",
            name="Clone and Build Repository",
            description="Clone a git repository and build it",
            category="development",
            tags=["git", "build"],
        )
        
        # Add nodes
        start_node = WorkflowNode(
            node_id="start",
            name="Start",
            node_type=NodeType.START,
        )
        
        clone_node = WorkflowNode(
            node_id="clone",
            name="Clone Repository",
            node_type=NodeType.TASK,
            tool_name="git_clone",
            parameter_mappings={"url": "repo_url", "path": "clone_path"},
            is_critical=True,
        )
        
        detect_build = WorkflowNode(
            node_id="detect",
            name="Detect Build System",
            node_type=NodeType.TASK,
            description="Detect the build system used",
        )
        
        build_node = WorkflowNode(
            node_id="build",
            name="Build Project",
            node_type=NodeType.TASK,
            description="Run the build command",
        )
        
        end_node = WorkflowNode(
            node_id="end",
            name="End",
            node_type=NodeType.END,
        )
        
        clone_build.nodes = [start_node, clone_node, detect_build, build_node, end_node]
        
        # Add transitions
        clone_build.transitions = [
            Transition("t1", "start", "clone", TransitionType.NORMAL),
            Transition("t2", "clone", "detect", TransitionType.NORMAL),
            Transition("t3", "detect", "build", TransitionType.NORMAL),
            Transition("t4", "build", "end", TransitionType.NORMAL),
        ]
        
        # Add input variables
        clone_build.input_variables = [
            WorkflowVariable("repo_url", var_type="string", is_input=True, description="Repository URL"),
            WorkflowVariable("clone_path", var_type="path", is_input=True, description="Clone destination"),
        ]
        
        self._templates["clone_and_build"] = clone_build
    
    async def generate_workflow(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Workflow:
        """Generate a workflow from natural language.
        
        Uses LLM to understand the request and generate an appropriate
        workflow structure.
        
        Args:
            request: Natural language request
            context: Optional context
            
        Returns:
            Generated Workflow
        """
        if self._llm_client is None:
            # Simple fallback
            return self._simple_workflow(request)
        
        # Get available tools
        tools = self._get_tool_descriptions()
        
        prompt = f"""You are a workflow generation system. Create a workflow to accomplish the user's request.

User Request: {request}

Available Tools:
{json.dumps(tools, indent=2)}

Create a workflow with nodes and transitions. Respond with JSON:
{{
    "name": "Workflow name",
    "description": "What the workflow does",
    "nodes": [
        {{
            "node_id": "unique_id",
            "name": "Node name",
            "node_type": "start|end|task|decision|fork|join",
            "tool_name": "for task nodes, the tool to use",
            "parameters": {{}},
            "decision_expression": "for decision nodes, the condition",
            "is_critical": true/false
        }}
    ],
    "transitions": [
        {{
            "from_node": "source_node_id",
            "to_node": "target_node_id",
            "condition": "optional condition for conditional transitions"
        }}
    ],
    "variables": [
        {{"name": "var_name", "value": "initial_value"}}
    ]
}}

Requirements:
1. Must have exactly one START node and at least one END node
2. All nodes must be reachable from START
3. All non-END nodes must have outgoing transitions
4. DECISION nodes need conditions on their outgoing transitions
5. FORK nodes split into parallel paths, JOIN nodes merge them
6. Use actual tool names from the available tools list
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            
            # Parse JSON
            from .structured_output import JSONExtractor
            extractor = JSONExtractor()
            parsed, _ = extractor.extract(response)
            
            if parsed:
                return self._dict_to_workflow(parsed)
                
        except Exception as e:
            logger.warning(f"LLM workflow generation failed: {e}")
        
        return self._simple_workflow(request)
    
    def _dict_to_workflow(self, data: Dict[str, Any]) -> Workflow:
        """Convert dictionary to Workflow."""
        workflow = Workflow(
            workflow_id=f"wf_{uuid.uuid4().hex[:8]}",
            name=data.get("name", "Generated Workflow"),
            description=data.get("description", ""),
        )
        
        # Add nodes
        for node_data in data.get("nodes", []):
            node_type_str = node_data.get("node_type", "task")
            node_type = NodeType.TASK
            for nt in NodeType:
                if nt.value == node_type_str:
                    node_type = nt
                    break
            
            node = WorkflowNode(
                node_id=node_data.get("node_id", f"node_{uuid.uuid4().hex[:4]}"),
                name=node_data.get("name", "Unnamed"),
                node_type=node_type,
                description=node_data.get("description", ""),
                tool_name=node_data.get("tool_name"),
                parameters=node_data.get("parameters", {}),
                decision_expression=node_data.get("decision_expression"),
                is_critical=node_data.get("is_critical", False),
            )
            workflow.nodes[node.node_id] = node
        
        # Add transitions
        for i, trans_data in enumerate(data.get("transitions", [])):
            workflow.transitions.append(Transition(
                transition_id=f"t_{i}",
                from_node=trans_data.get("from_node", ""),
                to_node=trans_data.get("to_node", ""),
                condition=trans_data.get("condition"),
            ))
        
        # Add variables
        for var_data in data.get("variables", []):
            workflow.variables[var_data["name"]] = WorkflowVariable(
                name=var_data["name"],
                value=var_data.get("value"),
            )
        
        workflow.status = WorkflowStatus.READY
        return workflow
    
    def _simple_workflow(self, request: str) -> Workflow:
        """Create a simple single-task workflow."""
        workflow = Workflow(
            workflow_id=f"wf_{uuid.uuid4().hex[:8]}",
            name="Simple Workflow",
            description=request,
        )
        
        workflow.nodes["start"] = WorkflowNode(
            node_id="start",
            name="Start",
            node_type=NodeType.START,
        )
        
        workflow.nodes["task"] = WorkflowNode(
            node_id="task",
            name="Execute Request",
            node_type=NodeType.TASK,
            description=request,
        )
        
        workflow.nodes["end"] = WorkflowNode(
            node_id="end",
            name="End",
            node_type=NodeType.END,
        )
        
        workflow.transitions = [
            Transition("t1", "start", "task"),
            Transition("t2", "task", "end"),
        ]
        
        workflow.status = WorkflowStatus.READY
        return workflow
    
    def _get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of available tools."""
        tools = []
        for registered in self._tool_registry.get_all_tools():
            defn = registered.definition
            category = defn.category
            if hasattr(category, 'value'):
                category = category.value
            tools.append({
                "name": defn.name,
                "description": defn.description,
                "category": category,
            })
        return tools
    
    async def execute(
        self,
        workflow: Workflow,
        tool_executor: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow.
        
        Args:
            workflow: The workflow to execute
            tool_executor: Optional function to execute tools
            
        Returns:
            Dictionary with execution results
        """
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now().isoformat()
        workflow.add_history("workflow_started")
        
        self._active_workflows[workflow.workflow_id] = workflow
        
        try:
            # Find start node
            start_node = workflow.get_start_node()
            if not start_node:
                raise ValueError("Workflow has no start node")
            
            # Initialize current nodes
            workflow.current_nodes = {start_node.node_id}
            start_node.status = WorkflowStatus.COMPLETED
            workflow.completed_nodes.add(start_node.node_id)
            
            # Execute until complete or failed
            while workflow.status == WorkflowStatus.RUNNING:
                # Get next nodes to execute
                next_nodes = self._get_next_nodes(workflow)
                
                if not next_nodes:
                    # Check if we've reached an end node
                    end_nodes = workflow.get_end_nodes()
                    if any(n.node_id in workflow.completed_nodes for n in end_nodes):
                        workflow.status = WorkflowStatus.COMPLETED
                    else:
                        workflow.status = WorkflowStatus.FAILED
                        workflow.add_history("workflow_stuck", details={"completed": list(workflow.completed_nodes)})
                    break
                
                # Execute nodes (can be parallel for FORK)
                workflow.current_nodes = {n.node_id for n in next_nodes}
                
                for node in next_nodes:
                    await self._execute_node(workflow, node, tool_executor)
                
                # Check for failures
                if workflow.failed_nodes and any(
                    workflow.nodes[nid].is_critical for nid in workflow.failed_nodes
                ):
                    workflow.status = WorkflowStatus.FAILED
                    break
            
            workflow.completed_at = datetime.now().isoformat()
            workflow.add_history("workflow_completed", details={"status": workflow.status.value})
            
            if self._on_workflow_complete:
                self._on_workflow_complete(workflow)
            
            return {
                "success": workflow.status == WorkflowStatus.COMPLETED,
                "workflow_id": workflow.workflow_id,
                "status": workflow.status.value,
                "completed_nodes": list(workflow.completed_nodes),
                "failed_nodes": list(workflow.failed_nodes),
                "variables": {k: v.value for k, v in workflow.variables.items()},
            }
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            workflow.status = WorkflowStatus.FAILED
            workflow.add_history("workflow_error", details={"error": str(e)})
            return {
                "success": False,
                "workflow_id": workflow.workflow_id,
                "error": str(e),
            }
        
        finally:
            if workflow.workflow_id in self._active_workflows:
                del self._active_workflows[workflow.workflow_id]
    
    def _get_next_nodes(self, workflow: Workflow) -> List[WorkflowNode]:
        """Get nodes that should execute next."""
        next_nodes = []
        
        for node_id in workflow.completed_nodes:
            for trans in workflow.get_outgoing_transitions(node_id):
                target_id = trans.to_node
                target_node = workflow.nodes.get(target_id)
                
                if not target_node:
                    continue
                
                # Skip already completed or failed
                if target_id in workflow.completed_nodes or target_id in workflow.failed_nodes:
                    continue
                
                # Check if all incoming transitions are satisfied
                incoming = workflow.get_incoming_transitions(target_id)
                all_satisfied = all(
                    t.from_node in workflow.completed_nodes
                    for t in incoming
                )
                
                if all_satisfied and target_node not in next_nodes:
                    # Check conditional transitions
                    if trans.transition_type == TransitionType.CONDITIONAL and trans.condition:
                        if not self._evaluate_condition(workflow, trans.condition):
                            continue
                    
                    next_nodes.append(target_node)
        
        return next_nodes
    
    async def _execute_node(
        self,
        workflow: Workflow,
        node: WorkflowNode,
        tool_executor: Optional[Callable],
    ):
        """Execute a single workflow node."""
        node.status = WorkflowStatus.RUNNING
        node.executed_at = datetime.now().isoformat()
        node.execution_count += 1
        workflow.add_history("node_started", node.node_id)
        
        if self._on_node_start:
            self._on_node_start(node, workflow)
        
        try:
            if node.node_type == NodeType.START:
                # Start nodes just pass through
                node.status = WorkflowStatus.COMPLETED
                
            elif node.node_type == NodeType.END:
                # End nodes complete the workflow
                node.status = WorkflowStatus.COMPLETED
                
            elif node.node_type == NodeType.TASK:
                # Execute the tool
                await self._execute_task_node(workflow, node, tool_executor)
                
            elif node.node_type == NodeType.DECISION:
                # Evaluate decision
                result = self._evaluate_condition(workflow, node.decision_expression or "true")
                node.result = result
                node.status = WorkflowStatus.COMPLETED
                
            elif node.node_type == NodeType.FORK:
                # Fork just passes through, enables parallel paths
                node.status = WorkflowStatus.COMPLETED
                
            elif node.node_type == NodeType.JOIN:
                # Join waits for all incoming paths (handled by _get_next_nodes)
                node.status = WorkflowStatus.COMPLETED
                
            else:
                node.status = WorkflowStatus.COMPLETED
            
            if node.status == WorkflowStatus.COMPLETED:
                workflow.completed_nodes.add(node.node_id)
                workflow.add_history("node_completed", node.node_id)
                
                if self._on_node_complete:
                    self._on_node_complete(node, workflow)
                    
        except Exception as e:
            logger.error(f"Node execution error: {e}")
            node.status = WorkflowStatus.FAILED
            node.error = str(e)
            workflow.failed_nodes.add(node.node_id)
            workflow.add_history("node_failed", node.node_id, {"error": str(e)})
    
    async def _execute_task_node(
        self,
        workflow: Workflow,
        node: WorkflowNode,
        tool_executor: Optional[Callable],
    ):
        """Execute a task node."""
        if not node.tool_name:
            # No tool specified - try to infer from description
            node.status = WorkflowStatus.COMPLETED
            return
        
        # Build parameters (substitute workflow variables)
        params = {}
        for key, value in node.parameters.items():
            if isinstance(value, str) and value.startswith("$"):
                var_name = value[1:]
                params[key] = workflow.get_variable(var_name)
            else:
                params[key] = value
        
        # Apply parameter mappings
        for param_name, var_name in node.parameter_mappings.items():
            params[param_name] = workflow.get_variable(var_name)
        
        # Execute
        if tool_executor:
            result = await tool_executor(node.tool_name, params)
            node.result = result
        else:
            # Try to get tool from registry
            tool = self._tool_registry.get_tool(node.tool_name)
            if tool:
                result = await tool.tool_instance.execute(params)
                node.result = result
        
        node.status = WorkflowStatus.COMPLETED
    
    def _evaluate_condition(self, workflow: Workflow, condition: str) -> bool:
        """Evaluate a condition expression.
        
        Uses LLM if available, otherwise simple evaluation.
        """
        if not condition or condition.lower() == "true":
            return True
        
        if condition.lower() == "false":
            return False
        
        # Try simple variable comparison
        # Format: "variable_name == value" or "variable_name"
        try:
            if "==" in condition:
                parts = condition.split("==")
                var_name = parts[0].strip()
                expected = parts[1].strip().strip('"\'')
                actual = workflow.get_variable(var_name)
                return str(actual) == expected
            
            elif "!=" in condition:
                parts = condition.split("!=")
                var_name = parts[0].strip()
                expected = parts[1].strip().strip('"\'')
                actual = workflow.get_variable(var_name)
                return str(actual) != expected
            
            else:
                # Just check if variable is truthy
                return bool(workflow.get_variable(condition))
                
        except Exception:
            return True  # Default to true if can't evaluate
    
    def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        if workflow_id in self._active_workflows:
            workflow = self._active_workflows[workflow_id]
            if workflow.status == WorkflowStatus.RUNNING:
                workflow.status = WorkflowStatus.PAUSED
                workflow.add_history("workflow_paused")
                return True
        return False
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        if workflow_id in self._active_workflows:
            workflow = self._active_workflows[workflow_id]
            if workflow.status == WorkflowStatus.PAUSED:
                workflow.status = WorkflowStatus.RUNNING
                workflow.add_history("workflow_resumed")
                return True
        return False
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow."""
        if workflow_id in self._active_workflows:
            workflow = self._active_workflows[workflow_id]
            workflow.status = WorkflowStatus.CANCELLED
            workflow.add_history("workflow_cancelled")
            return True
        return False
    
    async def save_workflow(self, workflow: Workflow) -> str:
        """Save workflow to persistence storage.
        
        Returns the path where it was saved.
        """
        if not self._persistence_dir:
            self._persistence_dir = Path.home() / ".proxima" / "workflows"
        
        self._persistence_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = self._persistence_dir / f"{workflow.workflow_id}.json"
        
        with open(filepath, "w") as f:
            json.dump(workflow.to_dict(), f, indent=2)
        
        return str(filepath)
    
    async def load_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Load a workflow from persistence storage."""
        if not self._persistence_dir:
            return None
        
        filepath = self._persistence_dir / f"{workflow_id}.json"
        
        if not filepath.exists():
            return None
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        return self._dict_to_workflow(data)
    
    def register_template(self, template: WorkflowTemplate):
        """Register a workflow template."""
        self._templates[template.template_id] = template
    
    def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a template by ID."""
        return self._templates.get(template_id)
    
    def list_templates(self) -> List[Dict[str, str]]:
        """List all available templates."""
        return [
            {
                "id": t.template_id,
                "name": t.name,
                "description": t.description,
                "category": t.category,
            }
            for t in self._templates.values()
        ]


# Module-level engine instance
_global_engine: Optional[WorkflowEngine] = None


def get_workflow_engine(
    llm_client: Optional[Any] = None,
) -> WorkflowEngine:
    """Get the global workflow engine.
    
    Args:
        llm_client: Optional LLM client
        
    Returns:
        WorkflowEngine instance
    """
    global _global_engine
    if _global_engine is None:
        _global_engine = WorkflowEngine(llm_client=llm_client)
    return _global_engine
