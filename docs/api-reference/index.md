# API Reference

Complete API documentation for the Proxima framework, auto-generated from source code docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

## Quick Links

| Module                        | Description                                         |
| ----------------------------- | --------------------------------------------------- |
| [Core](#core-modules)         | State machine, pipeline, planner, agent interpreter |
| [Backends](#backends)         | Quantum simulation adapters (Cirq, Qiskit, LRET)    |
| [Intelligence](#intelligence) | LLM routing, backend selection, insights            |
| [Resources](#resources)       | Resource monitoring, timer, consent, sessions       |
| [Data](#data)                 | Storage, comparison, export functionality           |
| [Config](#configuration)      | Configuration and settings management               |
| [CLI](#cli-reference)         | Command-line interface commands                     |

---

## Core Modules

### proxima.core.state

State machine for execution flow management.

::: proxima.core.state
options:
show_root_heading: true
show_source: false
members_order: source
heading_level: 4

### proxima.core.pipeline

Data flow pipeline orchestrator with async support.

::: proxima.core.pipeline
options:
show_root_heading: true
show_source: false
members: - PipelineStage - StageResult - PipelineContext - PipelineHandler - DataFlowPipeline - run_simulation - compare_backends
heading_level: 4

### proxima.core.planner

DAG-based execution planning.

::: proxima.core.planner
options:
show_root_heading: true
show_source: false
members: - TaskStatus - TaskNode - ExecutionDAG - Planner
heading_level: 4

### proxima.core.executor

Task execution engine.

::: proxima.core.executor
options:
show_root_heading: true
show_source: false
heading_level: 4

### proxima.core.agent_interpreter

Agent file parsing and execution.

::: proxima.core.agent_interpreter
options:
show_root_heading: true
show_source: false
members: - TaskDefinition - AgentInterpreter
heading_level: 4

---

## Backends

### proxima.backends.base

Base adapter interface and types.

::: proxima.backends.base
options:
show_root_heading: true
show_source: false
members_order: source
heading_level: 4

### proxima.backends.registry

Backend registry with hot-reload support.

::: proxima.backends.registry
options:
show_root_heading: true
show_source: false
members: - BackendRegistry - BackendStatus - backend_registry
heading_level: 4

### proxima.backends.lret

LRET backend adapter.

::: proxima.backends.lret
options:
show_root_heading: true
show_source: false
heading_level: 4

### proxima.backends.cirq_adapter

Google Cirq backend adapter.

::: proxima.backends.cirq_adapter
options:
show_root_heading: true
show_source: false
heading_level: 4

### proxima.backends.qiskit_adapter

IBM Qiskit Aer backend adapter.

::: proxima.backends.qiskit_adapter
options:
show_root_heading: true
show_source: false
heading_level: 4

---

## Intelligence

### proxima.intelligence.llm_router

LLM provider routing and management.

::: proxima.intelligence.llm_router
options:
show_root_heading: true
show_source: false
members_order: source
heading_level: 4

### proxima.intelligence.selector

Intelligent backend selection.

::: proxima.intelligence.selector
options:
show_root_heading: true
show_source: false
heading_level: 4

### proxima.intelligence.insights

Result analysis and insight generation.

::: proxima.intelligence.insights
options:
show_root_heading: true
show_source: false
heading_level: 4

---

## Resources

### proxima.resources.monitor

System resource monitoring.

::: proxima.resources.monitor
options:
show_root_heading: true
show_source: false
heading_level: 4

### proxima.resources.timer

Execution timing with ETA calculation.

::: proxima.resources.timer
options:
show_root_heading: true
show_source: false
members: - ETAResult - ETACalculator - ExecutionTimer
heading_level: 4

### proxima.resources.session

Session management with checkpoints.

::: proxima.resources.session
options:
show_root_heading: true
show_source: false
members: - Session - SessionManager
heading_level: 4

### proxima.resources.control

Execution flow control.

::: proxima.resources.control
options:
show_root_heading: true
show_source: false
heading_level: 4

### proxima.resources.consent

User consent management.

::: proxima.resources.consent
options:
show_root_heading: true
show_source: false
heading_level: 4

---

## Data

### proxima.data.store

Result storage and retrieval.

::: proxima.data.store
options:
show_root_heading: true
show_source: false
heading_level: 4

### proxima.data.compare

Multi-backend comparison engine.

::: proxima.data.compare
options:
show_root_heading: true
show_source: false
members: - ExecutionStrategy - BackendResult - ComparisonResult - BackendComparator
heading_level: 4

### proxima.data.export

Export engine for multiple formats.

::: proxima.data.export
options:
show_root_heading: true
show_source: false
members: - ExportFormat - ExportOptions - ReportData - ExportResult - CSVExporter - JSONExporter - XLSXExporter - HTMLExporter - MarkdownExporter - ExportEngine
heading_level: 4

---

## Configuration

### proxima.config

Configuration schema and management.

::: proxima.config
options:
show_root_heading: true
show_source: false
heading_level: 4

::: proxima.config.schema
options:
show_root_heading: true
show_source: false
heading_level: 4

---

## Utilities

### proxima.utils.logging

Structured logging utilities.

::: proxima.utils.logging
options:
show_root_heading: true
show_source: false
heading_level: 4

---

## CLI Reference

### Commands

| Command            | Description                      |
| ------------------ | -------------------------------- |
| `proxima init`     | Initialize Proxima configuration |
| `proxima config`   | Configuration management         |
| `proxima run`      | Execute a simulation             |
| `proxima compare`  | Multi-backend comparison         |
| `proxima backends` | Backend management               |
| `proxima history`  | Execution history                |
| `proxima session`  | Session management               |
| `proxima agent`    | Agent file operations            |
| `proxima version`  | Version information              |
| `proxima ui`       | Launch Terminal UI               |

### Global Options

| Option      | Short | Description          |
| ----------- | ----- | -------------------- |
| `--config`  | `-c`  | Path to config YAML  |
| `--backend` | `-b`  | Select backend       |
| `--output`  | `-o`  | Output format        |
| `--verbose` | `-v`  | Increase verbosity   |
| `--quiet`   | `-q`  | Decrease verbosity   |
| `--dry-run` |       | Plan only            |
| `--force`   | `-f`  | Skip consent prompts |

---

## Generate Locally

To generate and view API documentation locally:

```bash
# Install documentation dependencies
pip install -e .[docs]

# Serve documentation locally
mkdocs serve

# Build static documentation
mkdocs build
```
