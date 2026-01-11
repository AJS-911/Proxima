"""Core domain logic module - includes Step 5.2: Agent.md Interpreter."""

from .agent_interpreter import (
    # Enums
    TaskType,
    TaskStatus,
    ValidationSeverity,
    # Data classes
    ValidationIssue,
    AgentMetadata,
    AgentConfiguration,
    TaskDefinition,
    TaskResult,
    AgentFile,
    ExecutionReport,
    # Classes
    AgentFileParser,
    TaskExecutor,
    DefaultTaskExecutor,
    AgentInterpreter,
    # Convenience function
    run_agent_file,
)

from .executor import Executor

from .pipeline import (
    # Enums
    PipelineStage,
    # Data classes
    StageResult,
    PipelineContext,
    # Handler base class
    PipelineHandler,
    # Handler implementations
    ParseHandler,
    PlanHandler,
    ResourceCheckHandler,
    ConsentHandler,
    ExecutionHandler,
    CollectionHandler,
    AnalysisHandler,
    ExportHandler,
    # Main pipeline class
    DataFlowPipeline,
    # Convenience functions
    run_simulation,
    compare_backends,
)

from .planner import Planner

from .runner import quantum_runner

from .session import (
    # Enums
    SessionStatus,
    # Data classes
    Checkpoint,
    ExecutionRecord,
    # Classes
    Session,
    SessionManager,
)

from .state import (
    # Enums
    ExecutionState,
    # Classes
    ExecutionStateMachine,
)


__all__ = [
    # ============ agent_interpreter ============
    # Enums
    "TaskType",
    "TaskStatus",
    "ValidationSeverity",
    # Data classes
    "ValidationIssue",
    "AgentMetadata",
    "AgentConfiguration",
    "TaskDefinition",
    "TaskResult",
    "AgentFile",
    "ExecutionReport",
    # Classes
    "AgentFileParser",
    "TaskExecutor",
    "DefaultTaskExecutor",
    "AgentInterpreter",
    # Convenience function
    "run_agent_file",
    # ============ executor ============
    "Executor",
    # ============ pipeline ============
    # Enums
    "PipelineStage",
    # Data classes
    "StageResult",
    "PipelineContext",
    # Handler base class
    "PipelineHandler",
    # Handler implementations
    "ParseHandler",
    "PlanHandler",
    "ResourceCheckHandler",
    "ConsentHandler",
    "ExecutionHandler",
    "CollectionHandler",
    "AnalysisHandler",
    "ExportHandler",
    # Main pipeline class
    "DataFlowPipeline",
    # Convenience functions
    "run_simulation",
    "compare_backends",
    # ============ planner ============
    "Planner",
    # ============ runner ============
    "quantum_runner",
    # ============ session ============
    # Enums
    "SessionStatus",
    # Data classes
    "Checkpoint",
    "ExecutionRecord",
    # Classes
    "Session",
    "SessionManager",
    # ============ state ============
    # Enums
    "ExecutionState",
    # Classes
    "ExecutionStateMachine",
]
