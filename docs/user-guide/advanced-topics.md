# Advanced Topics

This guide covers advanced features and patterns for power users of Proxima.

## Table of Contents

- [Parallel Multi-Backend Execution](#parallel-multi-backend-execution)
- [DAG-Based Task Planning](#dag-based-task-planning)
- [Custom Pipeline Handlers](#custom-pipeline-handlers)
- [Session Checkpointing and Rollback](#session-checkpointing-and-rollback)
- [Advanced Export Options](#advanced-export-options)
- [Plugin Development](#plugin-development)
- [Performance Optimization](#performance-optimization)

---

## Parallel Multi-Backend Execution

Execute simulations across multiple backends simultaneously for comparison or redundancy.

### Basic Comparison

```python
from proxima.data.compare import BackendComparator, ExecutionStrategy

comparator = BackendComparator()

# Compare across backends with parallel execution
result = await comparator.compare(
    circuit=your_circuit,
    backends=["cirq", "qiskit-aer", "lret"],
    shots=10000,
    strategy=ExecutionStrategy.PARALLEL,
    max_concurrent=4,  # Limit parallel executions
    timeout_per_backend=60.0,  # Timeout per backend
)

# Analyze results
print(f"Fastest backend: {result.fastest_backend}")
print(f"Average fidelity: {result.average_fidelity:.4f}")
for backend, data in result.backend_results.items():
    print(f"  {backend}: {data.execution_time_ms:.1f}ms")
```

### Adaptive Strategy

The adaptive strategy automatically chooses parallel or sequential execution based on system resources:

```python
result = await comparator.compare(
    circuit=circuit,
    backends=backends,
    strategy=ExecutionStrategy.ADAPTIVE,
    memory_threshold_mb=4096,  # Switch to sequential below this
)
```

### Cancellation Support

```python
import asyncio

# Create cancellation event
cancel_event = asyncio.Event()

# Start comparison with cancellation support
task = asyncio.create_task(
    comparator.compare(
        circuit=circuit,
        backends=backends,
        cancellation_event=cancel_event,
    )
)

# Cancel if needed
await asyncio.sleep(30)
cancel_event.set()  # Signal cancellation
```

---

## DAG-Based Task Planning

Proxima uses Directed Acyclic Graphs (DAG) for optimal task scheduling.

### Creating a DAG Plan

```python
from proxima.core.planner import Planner, ExecutionDAG, TaskNode
from proxima.core.state import ExecutionStateMachine

# Create planner
sm = ExecutionStateMachine()
planner = Planner(sm)

# Generate DAG from objective
dag = planner.plan_as_dag("compare GHZ state across cirq and qiskit with 10000 shots")

# Inspect the DAG
print(f"Total tasks: {len(dag.nodes)}")
print(f"Execution levels: {len(dag.get_execution_order())}")
print(f"Critical path: {dag.get_critical_path()}")
print(f"Estimated duration: {dag.estimate_total_duration(max_parallel=4):.1f}ms")
```

### Custom Task Definitions

```python
from proxima.core.planner import TaskNode, TaskStatus

# Create custom tasks
task1 = TaskNode(
    action="initialize",
    description="Initialize quantum circuit",
    parameters={"qubits": 3, "type": "ghz"},
    priority=100,
    estimated_duration_ms=50.0,
    tags=["circuit", "init"],
)

task2 = TaskNode(
    action="execute",
    description="Execute on backend",
    parameters={"backend": "cirq", "shots": 1024},
    dependencies=[task1.task_id],  # Depends on task1
    priority=90,
    estimated_duration_ms=500.0,
)

# Add to DAG
dag = ExecutionDAG()
dag.add_task(task1)
dag.add_task(task2)

# Validate
is_valid, errors = dag.validate()
if not is_valid:
    print(f"Validation errors: {errors}")
```

### Execution Order Optimization

```python
# Get tasks grouped by level (parallel-friendly)
levels = dag.get_execution_order()
for i, level in enumerate(levels):
    print(f"Level {i}: {len(level)} tasks can run in parallel")
    for task_id in level:
        task = dag.nodes[task_id]
        print(f"  - {task.action}: {task.description}")
```

---

## Custom Pipeline Handlers

Extend the pipeline with custom stage handlers.

### Creating a Custom Handler

```python
from proxima.core.pipeline import (
    PipelineHandler,
    PipelineStage,
    StageResult,
    PipelineContext,
    DataFlowPipeline,
)

class CustomAnalysisHandler(PipelineHandler):
    """Custom analysis stage with ML integration."""

    def __init__(self, model_path: str = None):
        super().__init__(PipelineStage.ANALYZING)
        self.model_path = model_path

    async def execute(self, ctx: PipelineContext) -> StageResult:
        import time
        start = time.time()

        try:
            # Custom analysis logic
            insights = []
            for backend, result in ctx.backend_results.items():
                counts = result.get("counts", {})
                # Apply ML model or custom analysis
                prediction = self._analyze(counts)
                insights.append(f"{backend}: {prediction}")

            ctx.insights.extend(insights)

            return StageResult(
                stage=self.stage,
                success=True,
                data={"insights": insights},
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return StageResult(
                stage=self.stage,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _analyze(self, counts: dict) -> str:
        total = sum(counts.values())
        entropy = -sum(
            (v/total) * math.log2(v/total + 1e-10)
            for v in counts.values()
        )
        return f"entropy={entropy:.2f}"

# Use custom handler
pipeline = DataFlowPipeline()
pipeline.set_handler(PipelineStage.ANALYZING, CustomAnalysisHandler())
```

### Adding Callbacks

```python
# Sync callbacks
def on_stage_start(stage, ctx):
    print(f"Starting: {stage.name}")

def on_stage_complete(result, ctx):
    print(f"Completed: {result.stage.name} in {result.duration_ms:.1f}ms")

pipeline.on_stage_start(on_stage_start)
pipeline.on_stage_complete(on_stage_complete)

# Async callbacks
async def on_stage_start_async(stage, ctx):
    await log_to_database(stage.name, "started")

pipeline.on_stage_start_async(on_stage_start_async)
```

---

## Session Checkpointing and Rollback

Manage long-running sessions with checkpoint and recovery support.

### Creating Checkpoints

```python
from proxima.resources.session import SessionManager

# Create session manager
sm = SessionManager()
session = sm.create_session("my-experiment")

# Create checkpoint before risky operation
checkpoint_id = session.create_checkpoint()
print(f"Created checkpoint: {checkpoint_id}")

# Perform operation
try:
    result = await risky_operation()
except Exception as e:
    # Rollback on failure
    session.rollback_to(checkpoint_id)
    print("Rolled back to checkpoint")
```

### Session Metadata

```python
# Add metadata to session
session.add_metadata("experiment_name", "bell_state_comparison")
session.add_metadata("researcher", "Dr. Smith")

# Track execution
session.record_execution(
    backend="cirq",
    duration_ms=523.4,
    success=True,
    result_summary={"00": 512, "11": 512},
)

# Export session data
session_data = session.export()
```

---

## Advanced Export Options

### XLSX with Conditional Formatting

```python
from proxima.data.export import ExportEngine, ExportOptions, ExportFormat

engine = ExportEngine()
options = ExportOptions(
    format=ExportFormat.XLSX,
    output_path=Path("report.xlsx"),
    include_raw_results=True,
    include_comparison=True,
    include_insights=True,
    # XLSX-specific options
    xlsx_conditional_formatting=True,
    xlsx_include_charts=True,
    xlsx_freeze_panes=True,
)

result = engine.export(report_data, options)
```

### HTML with Custom Templates

```python
from jinja2 import Template

custom_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        /* Custom CSS */
        .highlight { background: yellow; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    {% for insight in insights %}
        <p class="highlight">{{ insight }}</p>
    {% endfor %}
</body>
</html>
"""

options = ExportOptions(
    format=ExportFormat.HTML,
    html_template=custom_template,
    html_inline_styles=True,
)
```

---

## Plugin Development

Create plugins to extend Proxima functionality.

### Plugin Structure

```python
# my_plugin/__init__.py
from proxima.plugins.base import ProximaPlugin

class MyAnalysisPlugin(ProximaPlugin):
    """Custom analysis plugin."""

    name = "my-analysis"
    version = "1.0.0"

    def on_load(self):
        """Called when plugin is loaded."""
        self.register_command("analyze", self.analyze_command)

    def on_execution_complete(self, ctx):
        """Hook called after execution."""
        # Add custom insights
        ctx.insights.append(self.custom_analysis(ctx.backend_results))

    def analyze_command(self, args):
        """Custom CLI command."""
        pass
```

### Registering Plugins

```python
from proxima.plugins import PluginManager

pm = PluginManager()
pm.load_plugin("my_plugin")
pm.enable("my-analysis")
```

---

## Performance Optimization

### Memory-Efficient Execution

```python
# Use memory threshold to control parallel execution
pipeline = DataFlowPipeline(memory_threshold_mb=2048)

# Enable garbage collection between backends
execution_handler = ExecutionHandler(
    enable_parallel=True,
    max_parallel=2,  # Limit parallelism
    timeout_per_backend=120.0,
)
pipeline.set_handler(PipelineStage.EXECUTING, execution_handler)
```

### Batch Processing

```python
# Process multiple circuits in batches
async def batch_execute(circuits, batch_size=10):
    results = []
    for i in range(0, len(circuits), batch_size):
        batch = circuits[i:i+batch_size]
        batch_results = await asyncio.gather(
            *[execute_circuit(c) for c in batch]
        )
        results.extend(batch_results)

        # Cleanup between batches
        import gc
        gc.collect()

    return results
```

### Caching Results

```python
from proxima.data.store import get_store

store = get_store()

# Check cache before execution
circuit_hash = hash_circuit(circuit)
cached = store.get_by_hash(circuit_hash)

if cached:
    return cached.results
else:
    result = await execute(circuit)
    store.save(result, circuit_hash=circuit_hash)
    return result
```

---

## Next Steps

- Explore the [API Reference](../api-reference/index.md) for detailed documentation
- See [Adding Backends](../developer-guide/adding-backends.md) to create custom adapters
- Check [Testing](../developer-guide/testing.md) for testing best practices
