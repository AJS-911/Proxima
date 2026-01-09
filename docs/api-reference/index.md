# API Reference

Complete API documentation for the Proxima framework.

## Core Modules

### proxima.core

Core execution engine, state management, and planning.

::: proxima.core.state
options:
show_root_heading: true
members_order: source

::: proxima.core.executor
options:
show_root_heading: true

::: proxima.core.planner
options:
show_root_heading: true

::: proxima.core.agent_interpreter
options:
show_root_heading: true

---

### proxima.backends

Quantum simulation backend adapters and registry.

::: proxima.backends.base
options:
show_root_heading: true
members_order: source

::: proxima.backends.registry
options:
show_root_heading: true

::: proxima.backends.cirq_adapter
options:
show_root_heading: true

::: proxima.backends.qiskit_adapter
options:
show_root_heading: true

---

### proxima.intelligence

LLM integration, backend selection, and insights.

::: proxima.intelligence.llm_router
options:
show_root_heading: true
members_order: source

::: proxima.intelligence.selector
options:
show_root_heading: true

::: proxima.intelligence.insights
options:
show_root_heading: true

---

### proxima.resources

Resource monitoring, execution control, and consent management.

::: proxima.resources.monitor
options:
show_root_heading: true

::: proxima.resources.timer
options:
show_root_heading: true

::: proxima.resources.control
options:
show_root_heading: true

::: proxima.resources.consent
options:
show_root_heading: true

---

### proxima.data

Data storage, comparison, and export functionality.

::: proxima.data.store
options:
show_root_heading: true

::: proxima.data.compare
options:
show_root_heading: true

::: proxima.data.export
options:
show_root_heading: true

---

### proxima.config

Configuration management and settings.

::: proxima.config.settings
options:
show_root_heading: true

---

### proxima.utils

Utility functions and helpers.

::: proxima.utils.logging
options:
show_root_heading: true

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
