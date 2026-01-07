# Proxima: Proper Implementation Steps

> **Document Type:** Theoretical High-Level Design (HLD)  
> **Version:** 2.0  
> **Date:** January 8, 2026  
> **Purpose:** Strategic, step-by-step, deeply detailed guide to build Proxima from scratch
> **Inspiration:** OpenCode AI, Crush (Charmbracelet) — architectural and UX patterns only

---

## Table of Contents

1. [Strategic System Sketch](#1-strategic-system-sketch)
   - Overall Architecture
   - Core Components
   - Data, Control, and Decision Flow
   - Feature Interconnection Map
2. [Phased Roadmap](#2-phased-roadmap)
3. [Phase-by-Phase Implementation Guide](#3-phase-by-phase-implementation-guide)
4. [Phase Summaries & Usage Guidance](#4-phase-summaries--usage-guidance)
5. [Appendix: Technology Stack](#appendix-technology-stack)

---

## Inspiration Sources (NOT COPYING)

Proxima draws architectural, UX, and feature inspiration from:

| Project                   | Repository                                                      | What We Learn                                             |
| ------------------------- | --------------------------------------------------------------- | --------------------------------------------------------- |
| **OpenCode AI**           | [opencode-ai/opencode](https://github.com/opencode-ai/opencode) | Agent orchestration, tool integration, context management |
| **Crush (Charmbracelet)** | [charmbracelet/crush](https://github.com/charmbracelet/crush)   | Terminal UI excellence, progress visualization, clean UX  |

**Proxima is designed as its own independent, extensible system—not a fork or derivative.**

---

## Mandatory Features Checklist

The following 11 features are **non-negotiable** requirements:

| #   | Feature                                               | Phase | Status |
| --- | ----------------------------------------------------- | ----- | ------ |
| 1   | Execution Timer & Transparency                        | 4     | 🔲     |
| 2   | Backend Selection & Intelligence                      | 2, 3  | 🔲     |
| 3   | Fail-Safe, Resource Awareness & Explicit Consent      | 4     | 🔲     |
| 4   | Execution Control (Start/Abort/Rollback/Pause/Resume) | 4     | 🔲     |
| 5   | Result Interpretation & Insights                      | 3, 5  | 🔲     |
| 6   | Multi-Backend Comparison                              | 5     | 🔲     |
| 7   | Planning, Analysis & Execution Pipeline               | 1, 2  | 🔲     |
| 8   | API Key & Local LLM Integration                       | 3     | 🔲     |
| 9   | proxima_agent.md Compatibility                        | 5     | 🔲     |
| 10  | Additional Features (OpenCode AI & Crush inspired)    | 5, 6  | 🔲     |
| 11  | UI (Future Work)                                      | 6     | 🔲     |

---

## Target Quantum Backends

| Backend        | Simulator Types              | Repository                                                                             |
| -------------- | ---------------------------- | -------------------------------------------------------------------------------------- |
| **LRET**       | Framework Integration        | [kunal5556/LRET](https://github.com/kunal5556/LRET/tree/feature/framework-integration) |
| **Cirq**       | Density Matrix, State Vector | [quantumlib/Cirq](https://github.com/quantumlib/Cirq)                                  |
| **Qiskit Aer** | Density Matrix, State Vector | [Qiskit/qiskit-aer](https://github.com/Qiskit/qiskit-aer)                              |
| **Extensible** | Custom plugins               | User-defined                                                                           |

---

## 1. Strategic System Sketch

### 1.1 Overall Architecture

Proxima follows a **layered modular architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: PRESENTATION                            │
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │     CLI     │    │     TUI     │    │   Web API   │            │
│   │   (Typer)   │    │  (Textual)  │    │  (FastAPI)  │            │
│   └─────────────┘    └─────────────┘    └─────────────┘            │
│         │                  │                  │                     │
│         └──────────────────┼──────────────────┘                     │
│                            │                                        │
│   ┌────────────────────────┴────────────────────────┐              │
│   │         Execution Timer & Progress Display       │  ← Feature 1│
│   └─────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 2: ORCHESTRATION                           │
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │   Planner   │◄──►│  Executor   │◄──►│    State    │            │
│   │  (Feature 7)│    │             │    │   Manager   │            │
│   └─────────────┘    └─────────────┘    └─────────────┘            │
│          │                  │                  │                    │
│          └──────────────────┼──────────────────┘                    │
│                             │                                       │
│   ┌─────────────────────────┴─────────────────────────┐            │
│   │    proxima_agent.md Interpreter (Feature 9)       │            │
│   └───────────────────────────────────────────────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 3: INTELLIGENCE                            │
│                                                                     │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│   │    LLM Router    │  │  Backend Selector │  │  Insight Engine  │ │
│   │ Local/Remote     │  │  (Auto-Selection) │  │  (Analysis)      │ │
│   │ (Feature 8)      │  │  (Feature 2)      │  │  (Feature 5)     │ │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 4: RESOURCES & SAFETY                      │
│                                                                     │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│   │  Memory Monitor  │  │  Execution Timer │  │  Consent Manager │ │
│   │  (Feature 3)     │  │  (Feature 1)     │  │  (Feature 3)     │ │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘ │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │   Execution Control: Start / Abort / Rollback / Pause /     │  │
│   │   Resume (Feature 4) — All state transitions visible        │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 5: BACKEND ABSTRACTION                     │
│                                                                     │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│   │    LRET    │  │    Cirq    │  │ Qiskit Aer │  │  Custom    │   │
│   │  Adapter   │  │  Adapter   │  │  Adapter   │  │  Plugins   │   │
│   │            │  │ (DM + SV)  │  │ (DM + SV)  │  │            │   │
│   └────────────┘  └────────────┘  └────────────┘  └────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 6: DATA & OUTPUT                           │
│                                                                     │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│   │   Result Store   │  │   Comparison     │  │  Export Engine   │ │
│   │   (JSON/SQLite)  │  │   Aggregator     │  │  (CSV/XLSX)      │ │
│   │                  │  │   (Feature 6)    │  │  (Feature 5)     │ │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component                 | Layer | Purpose                                                    | Mandatory Feature |
| ------------------------- | ----- | ---------------------------------------------------------- | ----------------- |
| **CLI/TUI**               | 1     | User interaction surface                                   | 11 (UI)           |
| **Execution Timer**       | 1, 4  | Display what's running, elapsed time, stage tracking       | 1                 |
| **Planner**               | 2     | Plan tasks, analyze requirements, create execution DAG     | 7                 |
| **Executor**              | 2     | Run tasks according to plan                                | 7                 |
| **State Manager**         | 2     | Track IDLE/PLANNING/RUNNING/PAUSED/ABORTED/COMPLETED/ERROR | 4                 |
| **Agent.md Interpreter**  | 2     | Parse and execute proxima_agent.md instructions            | 9                 |
| **LLM Router**            | 3     | Route to local or remote LLM with explicit consent         | 8                 |
| **Backend Selector**      | 3     | Auto-select optimal backend with explanation               | 2                 |
| **Insight Engine**        | 3     | Generate human-readable, analytical insights               | 5                 |
| **Memory Monitor**        | 4     | Detect low memory, RAM limits, hardware issues             | 3                 |
| **Consent Manager**       | 4     | Require explicit user consent; "force execute" option      | 3                 |
| **Execution Control**     | 4     | Start, Abort, Rollback, Pause, Resume                      | 4                 |
| **Backend Adapters**      | 5     | LRET, Cirq (DM/SV), Qiskit Aer (DM/SV), Custom             | 2                 |
| **Comparison Aggregator** | 6     | Multi-backend comparison with identical parameters         | 6                 |
| **Export Engine**         | 6     | Export to CSV, XLSX with insights (not raw dumps)          | 5                 |

### 1.3 Data, Control, and Decision Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CONTROL FLOW                                 │
│                                                                     │
│  User Command                                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   PLANNING  │───►│  RESOURCE   │───►│   CONSENT   │             │
│  │  (Feature 7)│    │    CHECK    │    │   GATE      │             │
│  └─────────────┘    │  (Feature 3)│    │  (Feature 3)│             │
│                     └─────────────┘    └──────┬──────┘             │
│                                               │                     │
│       ┌───────────────────────────────────────┘                     │
│       │                                                             │
│       ▼           User selects backend?                             │
│  ┌─────────────┐         │                                          │
│  │  BACKEND    │    ┌────┴────┐                                     │
│  │  SELECTION  │    │   Yes   │ → Use explicit choice               │
│  │  (Feature 2)│    └────┬────┘                                     │
│  └─────────────┘         │ No                                       │
│       │                  ▼                                          │
│       │         Auto-select + Explain why (Feature 2)               │
│       │                  │                                          │
│       └──────────────────┤                                          │
│                          ▼                                          │
│  ┌─────────────────────────────────────────────────────┐           │
│  │              EXECUTION (Feature 4)                  │           │
│  │  ┌─────────────────────────────────────────────┐   │           │
│  │  │  Timer Display: Task name + Elapsed + Stage │   │ Feature 1 │
│  │  └─────────────────────────────────────────────┘   │           │
│  │                                                     │           │
│  │  Controls: [Start] [Abort] [Pause] [Resume]        │           │
│  │            [Rollback if feasible]                  │           │
│  └───────────────────────────┬─────────────────────────┘           │
│                              │                                      │
│       ┌──────────────────────┼──────────────────────┐              │
│       ▼                      ▼                      ▼              │
│   Single Backend    OR   Multi-Backend Comparison (Feature 6)      │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────┐           │
│  │           RESULT INTERPRETATION (Feature 5)         │           │
│  │  • Human-readable insights                          │           │
│  │  • Analytical (not raw data dumps)                  │           │
│  │  • Decision-oriented recommendations                │           │
│  └───────────────────────────┬─────────────────────────┘           │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────┐           │
│  │              EXPORT (CSV/XLSX)                      │           │
│  └─────────────────────────────────────────────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Decision Points:**

| Decision Point    | Logic                                                                  |
| ----------------- | ---------------------------------------------------------------------- |
| Backend selection | User explicit → use that; else → auto-select and explain               |
| Resource check    | Insufficient → warn + require consent or abort                         |
| LLM invocation    | Local available → offer as option; require consent for local OR remote |
| Execution control | At any point: user can Abort, Pause, Resume                            |
| Rollback          | If feasible, restore to pre-execution state                            |

### 1.4 Feature Interconnection Map

```
Feature 1 (Timer) ─────────────────────┐
                                       │
Feature 2 (Backend Selection) ─────────┼───► Executor
                                       │
Feature 3 (Fail-Safe/Consent) ─────────┤
                                       │
Feature 4 (Execution Control) ─────────┤
                                       │
Feature 7 (Planning Pipeline) ─────────┘

Feature 5 (Insights) ◄───── Results from Feature 6 (Comparison)
                                       │
                                       ▼
Feature 8 (LLM) ───────────────► Insight Engine + Backend Selector
                                       │
                                       ▼
Feature 9 (agent.md) ──────────► Planner + Executor
                                       │
                                       ▼
Feature 10 (Inspired Features) ── Plugin system, session persistence, etc.
                                       │
                                       ▼
Feature 11 (UI) ───────────────► Presentation layer (future)
```

---

## 2. Phased Roadmap

### 2.1 Technology Stack

| Component         | Technology        | Purpose                  |
| ----------------- | ----------------- | ------------------------ |
| **Language**      | Python 3.11+      | Core implementation      |
| **CLI**           | Typer             | Command-line interface   |
| **TUI**           | Textual           | Terminal UI              |
| **Config**        | Pydantic Settings | Configuration management |
| **Async**         | asyncio           | Concurrent execution     |
| **Logging**       | Structlog         | Structured logging       |
| **Testing**       | pytest            | Test framework           |
| **State Machine** | transitions       | FSM implementation       |
| **Resources**     | psutil            | System monitoring        |
| **Keyring**       | keyring           | Secret storage           |
| **HTTP**          | httpx             | Async HTTP client        |
| **Data**          | Pandas + openpyxl | Data manipulation        |

### 2.2 Quantum Libraries

| Library    | Version                       | Simulator Types            |
| ---------- | ----------------------------- | -------------------------- |
| Cirq       | Latest stable                 | DensityMatrix, StateVector |
| Qiskit-Aer | Latest stable                 | AerSimulator (DM, SV)      |
| LRET       | feature/framework-integration | Custom                     |

### 2.3 LLM Integration

| Provider  | Library               | Notes              |
| --------- | --------------------- | ------------------ |
| OpenAI    | openai                | GPT-4, GPT-4-turbo |
| Anthropic | anthropic             | Claude models      |
| Ollama    | httpx (REST)          | Local inference    |
| LM Studio | httpx (OpenAI-compat) | Local inference    |

### 2.4 Phase Overview (27 Weeks Total)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       PROXIMA IMPLEMENTATION ROADMAP                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: FOUNDATION (Weeks 1-4)                                        │
│  ───────────────────────────────                                        │
│  • Project setup, CLI skeleton, config system                           │
│  • Logging, state machine, base architecture                            │
│  • Features: 1 (Timer), 4 (Execution Control basics)                    │
│                                                                          │
│  Phase 2: BACKENDS (Weeks 5-9)                                          │
│  ─────────────────────────────                                          │
│  • LRET, Cirq (DM + SV), Qiskit Aer (DM + SV) adapters                  │
│  • Backend registry and plugin system                                    │
│  • Features: 2 (Backend Selection)                                       │
│                                                                          │
│  Phase 3: INTELLIGENCE (Weeks 10-14)                                    │
│  ───────────────────────────────────                                    │
│  • LLM router (local + remote with consent)                             │
│  • Intelligent backend auto-selection with explanation                  │
│  • Insight engine for result interpretation                             │
│  • Features: 5 (Insights), 8 (LLM Integration)                          │
│                                                                          │
│  Phase 4: SAFETY (Weeks 15-18)                                          │
│  ─────────────────────────────                                          │
│  • Resource monitoring (RAM, CPU, low memory detection)                 │
│  • Consent manager (explicit consent, force execute option)             │
│  • Full execution control (Abort, Rollback, Pause, Resume)              │
│  • Features: 3 (Fail-Safe & Consent), 4 (Execution Control complete)    │
│                                                                          │
│  Phase 5: ADVANCED (Weeks 19-23)                                        │
│  ──────────────────────────────                                         │
│  • Multi-backend comparison with identical parameters                   │
│  • proxima_agent.md parsing and execution                               │
│  • Export engine (CSV, XLSX with insights)                              │
│  • Features: 6 (Multi-Backend Comparison), 7 (Planning Pipeline),       │
│              9 (agent.md), 10 (Inspired Features)                       │
│                                                                          │
│  Phase 6: PRODUCTION (Weeks 24-27)                                      │
│  ─────────────────────────────────                                      │
│  • TUI implementation with Textual                                      │
│  • Comprehensive testing (unit, integration, e2e)                       │
│  • Documentation and packaging                                          │
│  • Features: 11 (UI)                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.5 Feature-to-Phase Mapping

| Feature # | Feature Name                            | Primary Phase    | Secondary Phases   |
| --------- | --------------------------------------- | ---------------- | ------------------ |
| 1         | Execution Timer & Transparency          | Phase 1          | Phase 4            |
| 2         | Backend Selection & Intelligence        | Phase 2          | Phase 3            |
| 3         | Fail-Safe, Resource Awareness & Consent | Phase 4          | Phase 1            |
| 4         | Execution Control                       | Phase 1 (basics) | Phase 4 (complete) |
| 5         | Result Interpretation & Insights        | Phase 3          | Phase 5            |
| 6         | Multi-Backend Comparison                | Phase 5          | -                  |
| 7         | Planning, Analysis & Execution Pipeline | Phase 5          | Phase 1            |
| 8         | API Key & Local LLM Integration         | Phase 3          | -                  |
| 9         | proxima_agent.md Compatibility          | Phase 5          | -                  |
| 10        | Additional Inspired Features            | Phase 5          | All Phases         |
| 11        | UI (TUI/Web)                            | Phase 6          | -                  |

---

## 3. Phase-by-Phase Implementation Guide

---

### PHASE 1: Foundation & Core Infrastructure

**Duration:** 3-4 weeks  
**Goal:** Establish project skeleton and core systems  
**Mandatory Features Addressed:** 1 (Timer basics), 4 (Execution Control basics)

---

#### Step 1.1: Project Structure Setup

**Create Directory Structure:**

```
proxima/
├── pyproject.toml          # Project metadata and dependencies
├── README.md               # Project documentation
├── LICENSE                 # License file
├── .gitignore              # Git ignore rules
├── .env.example            # Environment variable template
│
├── src/
│   └── proxima/
│       ├── __init__.py     # Package initialization
│       ├── __main__.py     # Entry point for `python -m proxima`
│       ├── cli/            # Command-line interface
│       │   ├── __init__.py
│       │   ├── main.py     # CLI app definition
│       │   ├── commands/   # Individual commands
│       │   │   ├── __init__.py
│       │   │   ├── run.py
│       │   │   ├── config.py
│       │   │   ├── backends.py
│       │   │   └── compare.py
│       │   └── utils.py    # CLI utilities
│       │
│       ├── core/           # Core domain logic
│       │   ├── __init__.py
│       │   ├── state.py    # State machine
│       │   ├── planner.py  # Execution planner
│       │   ├── executor.py # Task executor
│       │   └── session.py  # Session management
│       │
│       ├── backends/       # Backend adapters
│       │   ├── __init__.py
│       │   ├── base.py     # Abstract base adapter
│       │   ├── registry.py # Backend registry
│       │   ├── lret.py     # LRET adapter
│       │   ├── cirq_adapter.py  # Cirq adapter
│       │   └── qiskit_adapter.py # Qiskit adapter
│       │
│       ├── intelligence/   # AI/ML components
│       │   ├── __init__.py
│       │   ├── llm_router.py    # LLM routing logic
│       │   ├── selector.py      # Backend auto-selection
│       │   └── insights.py      # Result interpretation
│       │
│       ├── resources/      # Resource management
│       │   ├── __init__.py
│       │   ├── monitor.py  # Memory/CPU monitoring
│       │   ├── timer.py    # Execution timing
│       │   ├── consent.py  # Consent management
│       │   └── control.py  # Execution control
│       │
│       ├── data/           # Data handling
│       │   ├── __init__.py
│       │   ├── store.py    # Result storage
│       │   ├── compare.py  # Comparison aggregator
│       │   └── export.py   # Export engine
│       │
│       ├── config/         # Configuration
│       │   ├── __init__.py
│       │   ├── settings.py # Pydantic settings
│       │   └── defaults.py # Default values
│       │
│       └── utils/          # Shared utilities
│           ├── __init__.py
│           ├── logging.py  # Logging setup
│           └── helpers.py  # General helpers
│
├── tests/                  # Test suites
│   ├── __init__.py
│   ├── conftest.py         # Pytest fixtures
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-end tests
│
├── configs/                # Configuration files
│   └── default.yaml        # Default configuration
│
└── docs/                   # Documentation
    ├── user-guide/
    ├── developer-guide/
    └── api-reference/
```

**Dependencies to Install:**

- Core: typer, pydantic, pydantic-settings, structlog
- Async: asyncio (stdlib), anyio
- Monitoring: psutil
- Testing: pytest, pytest-asyncio, pytest-mock
- Development: black, ruff, mypy

---

#### Step 1.2: Configuration System

**Configuration Hierarchy (Priority Order):**

1. Command-line arguments (highest)
2. Environment variables (PROXIMA\_\*)
3. User config file (~/.proxima/config.yaml)
4. Project config file (./proxima.yaml)
5. Default values (lowest)

**Configuration Categories:**

```
Settings:
├── General
│   ├── verbosity: debug | info | warning | error
│   ├── output_format: text | json | rich
│   └── color_enabled: boolean
│
├── Backends
│   ├── default_backend: auto | lret | cirq | qiskit
│   ├── parallel_execution: boolean
│   └── timeout_seconds: integer
│
├── LLM
│   ├── provider: openai | anthropic | local | none
│   ├── model: string (e.g., gpt-4)
│   ├── local_endpoint: URL
│   ├── api_key_env_var: string
│   └── require_consent: boolean
│
├── Resources
│   ├── memory_warn_threshold_mb: integer
│   ├── memory_critical_threshold_mb: integer
│   └── max_execution_time_seconds: integer
│
└── Consent
    ├── auto_approve_local_llm: boolean
    ├── auto_approve_remote_llm: boolean
    └── remember_decisions: boolean
```

**Implementation Approach:**

1. Define Pydantic Settings class with nested models
2. Create config loader that merges sources in priority order
3. Implement config validation with clear error messages
4. Add CLI commands for config view/edit
5. Support config export/import for portability

---

#### Step 1.3: Logging Infrastructure

**Logging Architecture:**

```
Log Events → Structlog Processors → Output Handlers
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
              Console Output         File Output         JSON Output
              (Rich formatting)      (Plain text)        (Structured)
```

**Log Levels and Usage:**

| Level   | Usage                        | Example                                   |
| ------- | ---------------------------- | ----------------------------------------- |
| DEBUG   | Detailed execution trace     | "Entering backend.execute with params..." |
| INFO    | Normal operations            | "Simulation completed in 2.3s"            |
| WARNING | Resource concerns, consent   | "Memory usage at 85%"                     |
| ERROR   | Failures requiring attention | "Backend execution failed"                |

**Structured Log Fields:**

- timestamp: ISO 8601 format
- level: Log level
- event: Description
- component: Source component (e.g., "backend.cirq")
- execution_id: Unique execution identifier
- duration_ms: For timed operations
- metadata: Additional context

---

#### Step 1.4: State Machine Implementation

**States:**

```
┌────────┐
│  IDLE  │ ◄─────────────────────────────────────────┐
└────┬───┘                                           │
     │ start                                         │
     ▼                                               │
┌──────────┐                                         │
│ PLANNING │                                         │
└────┬─────┘                                         │
     │ plan_complete                                 │
     ▼                                               │
┌─────────┐                                          │
│  READY  │                                          │
└────┬────┘                                          │
     │ execute                                       │
     ▼                                               │
┌─────────┐    pause    ┌────────┐                  │
│ RUNNING │ ──────────► │ PAUSED │                  │
└────┬────┘             └────┬───┘                  │
     │                       │ resume                │
     │ ◄─────────────────────┘                      │
     │                                               │
     ├── complete ──────► ┌───────────┐ ───────────►┤
     │                    │ COMPLETED │              │
     │                    └───────────┘              │
     │                                               │
     ├── abort ─────────► ┌─────────┐ ─────────────►┤
     │                    │ ABORTED │                │
     │                    └─────────┘                │
     │                                               │
     └── error ─────────► ┌───────┐ ───────────────►┘
                          │ ERROR │
                          └───────┘
```

**State Transition Rules:**

| From     | To        | Trigger         | Conditions                         |
| -------- | --------- | --------------- | ---------------------------------- |
| IDLE     | PLANNING  | start()         | Valid input provided               |
| PLANNING | READY     | plan_complete() | Plan validated                     |
| PLANNING | ERROR     | plan_failed()   | Invalid plan                       |
| READY    | RUNNING   | execute()       | Resources available, consent given |
| RUNNING  | PAUSED    | pause()         | At checkpoint                      |
| RUNNING  | COMPLETED | complete()      | Execution successful               |
| RUNNING  | ABORTED   | abort()         | User request or critical error     |
| RUNNING  | ERROR     | error()         | Unrecoverable failure              |
| PAUSED   | RUNNING   | resume()        | State restored                     |
| PAUSED   | ABORTED   | abort()         | User request                       |
| \*       | IDLE      | reset()         | User request, cleanup complete     |

**Implementation:**

- Use `transitions` library for FSM
- Add callbacks for state entry/exit
- Log all transitions with timestamps
- Persist state for recovery

---

#### Step 1.5: CLI Scaffold

**Command Structure:**

```
proxima
├── init              # Initialize configuration
├── config            # Configuration management
│   ├── show          # Display current config
│   ├── set           # Set a value
│   ├── get           # Get a value
│   └── reset         # Reset to defaults
├── run               # Execute simulation
├── compare           # Multi-backend comparison
├── backends          # Backend management
│   ├── list          # List available
│   ├── info          # Show details
│   └── test          # Test connectivity
├── history           # Execution history
├── session           # Session management
│   ├── list          # List sessions
│   └── resume        # Resume session
├── agent             # Agent.md operations
│   └── run           # Execute agent file
├── version           # Version info
└── ui                # Launch TUI (Phase 6)
```

**Global Flags:**

| Flag      | Short | Description                         |
| --------- | ----- | ----------------------------------- |
| --verbose | -v    | Increase verbosity (stackable)      |
| --quiet   | -q    | Suppress non-essential output       |
| --config  | -c    | Specify config file                 |
| --backend | -b    | Specify backend                     |
| --dry-run |       | Show plan without executing         |
| --force   | -f    | Skip consent prompts (with warning) |
| --output  | -o    | Output format (text/json/table)     |

---

### PHASE 2: Backend Integration & Abstraction

**Duration:** 4-5 weeks  
**Goal:** Create unified interface for quantum backends

---

#### Step 2.1: Backend Interface Definition

**Abstract Base Class:**

Define interface that all adapters must implement:

| Method                        | Returns          | Purpose                  |
| ----------------------------- | ---------------- | ------------------------ |
| `get_name()`                  | str              | Backend identifier       |
| `get_version()`               | str              | Backend version          |
| `get_capabilities()`          | Capabilities     | Feature flags and limits |
| `validate_circuit(circuit)`   | ValidationResult | Check compatibility      |
| `estimate_resources(circuit)` | ResourceEstimate | Memory/time estimates    |
| `execute(circuit, options)`   | ExecutionResult  | Run simulation           |
| `supports_simulator(type)`    | bool             | Check simulator support  |

**Capability Model:**

```
Capabilities:
├── simulator_types: List[SimulatorType]
│   ├── STATE_VECTOR
│   ├── DENSITY_MATRIX
│   └── CUSTOM
├── max_qubits: int
├── supports_noise: bool
├── supports_gpu: bool
├── supports_batching: bool
└── custom_features: Dict[str, Any]
```

**Result Model:**

```
ExecutionResult:
├── backend: str
├── simulator_type: SimulatorType
├── execution_time_ms: float
├── qubit_count: int
├── shot_count: Optional[int]
├── result_type: ResultType
│   ├── COUNTS
│   ├── STATEVECTOR
│   └── DENSITY_MATRIX
├── data: ResultData
│   ├── counts: Optional[Dict[str, int]]
│   ├── statevector: Optional[ndarray]
│   └── density_matrix: Optional[ndarray]
├── metadata: Dict[str, Any]
└── raw_result: Any  # Original backend result
```

---

#### Step 2.2: Backend Registry

**Registry Responsibilities:**

1. Discover installed backends
2. Maintain adapter instances
3. Report backend health/availability
4. Support dynamic registration

**Discovery Process:**

```
1. Check for installed packages (cirq, qiskit-aer, lret)
2. Import and instantiate adapters for found packages
3. Validate each adapter can initialize
4. Cache capabilities for quick lookup
5. Mark unavailable backends with reason
```

**Registry Interface:**

| Method                   | Purpose                     |
| ------------------------ | --------------------------- |
| `discover()`             | Scan for available backends |
| `get(name)`              | Get adapter by name         |
| `list_available()`       | List working backends       |
| `get_capabilities(name)` | Get backend capabilities    |
| `is_available(name)`     | Check if backend usable     |

---

#### Step 2.3: LRET Adapter

**Integration Steps:**

1. Add LRET as git dependency (feature/framework-integration branch)
2. Create adapter class extending base
3. Implement circuit translation if needed
4. Map LRET API to unified interface
5. Handle LRET-specific result formats
6. Implement error mapping

**LRET-Specific Considerations:**

- Framework integration branch may have different API
- Custom simulation modes need mapping
- Result normalization to standard format

---

#### Step 2.4: Cirq Adapter

**Integration Steps:**

1. Add cirq as dependency
2. Create adapter with dual simulator support
3. Implement State Vector path using cirq.Simulator
4. Implement Density Matrix path using cirq.DensityMatrixSimulator
5. Handle Cirq's Moment and Circuit structures
6. Normalize measurement results

**Simulator Selection Logic:**

| Condition                 | Simulator     | Reason                |
| ------------------------- | ------------- | --------------------- |
| Noise model required      | DensityMatrix | Supports mixed states |
| Pure state, large circuit | StateVector   | More memory efficient |
| Small circuit, no noise   | StateVector   | Faster execution      |
| User specifies            | As specified  | User preference       |

---

#### Step 2.5: Qiskit Aer Adapter

**Integration Steps:**

1. Add qiskit-aer as dependency
2. Create adapter using AerSimulator
3. Configure for statevector mode
4. Configure for density_matrix mode
5. Support Qiskit's noise models
6. Handle Qiskit's job execution pattern

**Qiskit-Specific Features:**

- Transpilation before execution
- Backend options for simulator configuration
- Shot-based execution for counts
- Snapshot-based execution for statevector

---

#### Step 2.6: Result Normalization

**Normalization Pipeline:**

```
Backend Raw Result
        │
        ▼
┌───────────────────┐
│ Extract Core Data │ ← Pull relevant fields
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Convert Formats   │ ← Standardize types (numpy arrays, etc.)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Calculate Metrics │ ← Execution time, fidelity, etc.
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Build Result Obj  │ ← Populate ExecutionResult
└─────────┬─────────┘
          │
          ▼
    ExecutionResult
```

**Probability Normalization:**

- Ensure probabilities sum to 1.0
- Handle floating-point precision
- Sort states consistently (little-endian/big-endian)
- Convert between different state labeling conventions

---

### PHASE 3: Intelligence & Decision Systems

**Duration:** 4-5 weeks  
**Goal:** Add LLM capabilities and intelligent automation

---

#### Step 3.1: LLM Router Architecture

**Router Components:**

```
LLMRouter
├── ProviderRegistry
│   ├── OpenAIProvider
│   ├── AnthropicProvider
│   ├── OllamaProvider
│   └── LMStudioProvider
│
├── LocalLLMDetector
│   ├── check_ollama()
│   ├── check_lm_studio()
│   └── scan_model_files()
│
├── APIKeyManager
│   ├── store_key()
│   ├── get_key()
│   └── validate_key()
│
└── ConsentGate
    ├── request_consent()
    ├── check_consent()
    └── remember_consent()
```

**Request Routing Logic:**

```
1. Parse request to determine LLM need
2. Check if user consented to LLM use
3. If local LLM available and consented → use local
4. If remote API available and consented → use remote
5. If neither → proceed without LLM or prompt user
```

---

#### Step 3.2: Local LLM Detection

**Detection Methods:**

| Runtime          | Detection Method  | Default Port |
| ---------------- | ----------------- | ------------ |
| Ollama           | HTTP health check | 11434        |
| LM Studio        | HTTP health check | 1234         |
| llama.cpp server | HTTP health check | 8080         |
| Model files      | Directory scan    | N/A          |

**Detection Flow:**

```
1. Check configured local_endpoint first
2. Try default ports for known runtimes
3. Verify model availability
4. Cache detection results
5. Re-detect on user request or failure
```

---

#### Step 3.3: Backend Auto-Selection

**Selection Algorithm:**

```
Input: Circuit, UserPreferences, AvailableBackends

1. EXTRACT circuit characteristics:
   - qubit_count
   - gate_types (list of gates used)
   - circuit_depth
   - has_measurements
   - needs_noise

2. FOR each available backend:
   - Check if supports required features
   - Calculate compatibility score
   - Estimate execution time
   - Estimate memory usage
   - Check resource availability

3. RANK backends by:
   - Feature compatibility (must-have)
   - Performance score (nice-to-have)
   - Resource efficiency

4. SELECT top-ranked backend

5. GENERATE explanation:
   - Why this backend chosen
   - Trade-offs considered
   - Alternatives available

6. RETURN (selected_backend, explanation)
```

**Scoring Weights:**

| Factor            | Weight | Description                     |
| ----------------- | ------ | ------------------------------- |
| Feature Match     | 0.4    | Required features supported     |
| Performance       | 0.3    | Historical execution speed      |
| Memory Efficiency | 0.2    | Lower memory preference         |
| User History      | 0.1    | Previously successful with user |

---

#### Step 3.4: Insight Engine

**Analysis Pipeline:**

```
Raw Results
     │
     ▼
┌─────────────────┐
│ Statistical     │ ← Mean, variance, entropy
│ Analysis        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pattern         │ ← Dominant states, anomalies
│ Detection       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Synthesis   │ ← Natural language (if consented)
│ (Optional)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Insight         │ ← Structured insight object
│ Formatting      │
└────────┬────────┘
         │
         ▼
    InsightReport
```

**Insight Categories:**

1. **Summary:** One-paragraph overview
2. **Key Findings:** Bullet points of significant observations
3. **Statistical Metrics:** Quantitative analysis
4. **Recommendations:** Suggested next steps
5. **Visualizations:** Chart suggestions or ASCII representations

---

### PHASE 4: Safety, Control & Transparency

**Duration:** 3-4 weeks  
**Goal:** Implement resource monitoring and execution control

---

#### Step 4.1: Memory Monitoring

**Monitoring Architecture:**

```
MemoryMonitor
├── Continuous monitoring thread/task
├── Threshold configuration
├── Alert callbacks
└── History tracking

Thresholds:
├── INFO: 60% of available
├── WARNING: 80% of available
├── CRITICAL: 95% of available
└── ABORT: Out of memory imminent
```

**Pre-Execution Check:**

```
1. Get current available memory
2. Estimate simulation requirement
3. Compare requirement vs available
4. If insufficient:
   a. Calculate shortfall
   b. Generate warning message
   c. Request consent to proceed or abort
5. If sufficient: proceed with monitoring
```

---

#### Step 4.2: Execution Timer

**Timer Components:**

```
ExecutionTimer
├── GlobalTimer
│   └── Total elapsed since start
├── StageTimer
│   └── Per-stage elapsed times
├── ETACalculator
│   └── Estimated time remaining
└── ProgressTracker
    └── Percentage completion
```

**Display Update Strategy:**

- Update every 100ms for active stages
- Update on stage transitions
- Update on significant progress (10% increments)
- Batch updates to avoid flicker

---

#### Step 4.3: Execution Control

**Control Implementation:**

| Operation  | Mechanism                                      |
| ---------- | ---------------------------------------------- |
| **Start**  | Initialize state, begin execution loop         |
| **Abort**  | Set abort flag, cleanup, transition to ABORTED |
| **Pause**  | Set pause flag, checkpoint state, wait         |
| **Resume** | Clear pause flag, restore, continue            |

**Checkpoint Strategy:**

- Define safe checkpoint locations (between stages)
- At checkpoint: serialize state to temporary file
- On resume: load checkpoint, validate, continue
- Clean up checkpoints on completion

---

#### Step 4.4: Consent Management

**Consent Flow:**

```
Action Requested
       │
       ▼
┌──────────────────┐
│ Check Remembered │
└────────┬─────────┘
         │
    ┌────┴────┐
    │ Found?  │
    └────┬────┘
    yes  │  no
    ┌────┴────┐
    │         │
    ▼         ▼
 Proceed   ┌──────────────────┐
           │ Display Consent  │
           │ Prompt           │
           └────────┬─────────┘
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
       Approve   Remember    Deny
         │          │          │
         ▼          ▼          ▼
      Proceed   Save &     Return
               Proceed    Error
```

**Consent Categories:**

| Category           | Remember Option         | Force Override |
| ------------------ | ----------------------- | -------------- |
| Local LLM          | Yes (session/permanent) | Yes            |
| Remote LLM         | Yes (session/permanent) | Yes            |
| Force Execute      | No (always ask)         | N/A            |
| Untrusted agent.md | No (always ask)         | No             |

---

### PHASE 5: Advanced Features

**Duration:** 4-5 weeks  
**Goal:** Multi-backend comparison, agent.md, advanced exports

---

#### Step 5.1: Multi-Backend Comparison

**Comparison Workflow:**

```
1. User specifies backends to compare
2. Validate circuit on all backends
3. Plan parallel execution (if resources allow)
4. Execute on each backend with same parameters
5. Collect and normalize results
6. Calculate comparison metrics
7. Generate comparison report
```

**Parallel Execution Strategy:**

```
IF sum(memory_requirements) < available_memory * 0.8:
    Execute in parallel using asyncio.gather()
ELSE:
    Execute sequentially with cleanup between
```

**Comparison Metrics:**

| Metric            | Description                 |
| ----------------- | --------------------------- |
| Execution Time    | Wall-clock time per backend |
| Memory Peak       | Maximum memory usage        |
| Result Agreement  | Percentage similarity       |
| Fidelity          | For statevector comparisons |
| Performance Ratio | Time ratio between backends |

---

#### Step 5.2: Agent.md Interpreter

**File Parser:**

```
1. Read file content
2. Parse as Markdown
3. Extract metadata section
4. Extract configuration section
5. Parse task definitions
6. Validate task parameters
7. Build execution plan
```

**Task Execution:**

```
FOR each task in agent_file.tasks:
    1. Display task description
    2. Request consent for sensitive operations
    3. Create task execution plan
    4. Execute using standard pipeline
    5. Collect results
    6. Continue to next task or stop on error

FINALLY:
    Generate combined report
```

---

#### Step 5.3: Export Engine

**Export Formats:**

| Format | Library       | Features                    |
| ------ | ------------- | --------------------------- |
| CSV    | csv (stdlib)  | Simple tabular data         |
| XLSX   | openpyxl      | Multiple sheets, formatting |
| JSON   | json (stdlib) | Full data structure         |
| HTML   | jinja2        | Rich formatted reports      |

**Report Structure (XLSX):**

```
Workbook:
├── Sheet: Summary
│   └── Overview, key metrics
├── Sheet: Raw Results
│   └── Full measurement data
├── Sheet: Backend Comparison
│   └── Side-by-side metrics
├── Sheet: Insights
│   └── Generated insights
└── Sheet: Metadata
    └── Execution details
```

---

### PHASE 6: Production Hardening

**Duration:** 3-4 weeks  
**Goal:** TUI, testing, documentation, packaging

---

#### Step 6.1: Terminal UI

**TUI Framework:** Textual (Python)

**Screens:**

1. **Dashboard:** System status, recent executions
2. **Execution:** Real-time progress, logs
3. **Configuration:** Settings management
4. **Results:** Browse and analyze results
5. **Backends:** Backend status and management

**Design Principles:**

- Keyboard-first navigation
- Responsive to terminal size
- Consistent color theme
- Contextual help (press ? for help)

---

#### Step 6.2: Testing Strategy

**Test Pyramid:**

```
        ┌───────────┐
        │   E2E     │  10%
        └─────┬─────┘
        ┌─────┴─────┐
        │Integration│  30%
        └─────┬─────┘
    ┌─────────┴─────────┐
    │      Unit         │  60%
    └───────────────────┘
```

**Test Categories:**

| Category    | Focus                 | Tools              |
| ----------- | --------------------- | ------------------ |
| Unit        | Individual functions  | pytest, mock       |
| Integration | Component interaction | pytest, fixtures   |
| Backend     | Adapter functionality | Mock backends      |
| E2E         | Full workflows        | pytest, CLI runner |
| Performance | Resource usage        | pytest-benchmark   |

---

#### Step 6.3: Documentation

**Documentation Structure:**

```
docs/
├── index.md              # Home
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   └── configuration.md
├── user-guide/
│   ├── running-simulations.md
│   ├── comparing-backends.md
│   ├── using-llm.md
│   └── agent-files.md
├── developer-guide/
│   ├── architecture.md
│   ├── adding-backends.md
│   ├── contributing.md
│   └── testing.md
└── api-reference/
    └── [auto-generated]
```

**Documentation Tools:**

- MkDocs with Material theme
- mkdocstrings for API docs
- GitHub Pages for hosting

---

#### Step 6.4: Packaging

**Distribution Channels:**

| Channel  | Command                     | Notes         |
| -------- | --------------------------- | ------------- |
| PyPI     | `pip install proxima-agent` | Primary       |
| Homebrew | `brew install proxima`      | macOS/Linux   |
| Binaries | Download from releases      | All platforms |
| Docker   | `docker run proxima`        | Containerized |

**Release Checklist:**

1. Update version number
2. Update changelog
3. Run full test suite
4. Build packages
5. Test installation
6. Tag release
7. Publish to channels
8. Update documentation

---

## Integration Points

### External System Integration

```
┌─────────────────────────────────────────────────────────────┐
│                        PROXIMA                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐ │
│  │   Quantum   │      │    LLM      │      │   System    │ │
│  │  Libraries  │      │ Providers   │      │  Resources  │ │
│  └──────┬──────┘      └──────┬──────┘      └──────┬──────┘ │
│         │                    │                    │         │
└─────────┼────────────────────┼────────────────────┼─────────┘
          │                    │                    │
          ▼                    ▼                    ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │    Cirq     │     │   OpenAI    │     │   psutil    │
    │   Qiskit    │     │  Anthropic  │     │  (Memory)   │
    │    LRET     │     │   Ollama    │     │             │
    └─────────────┘     └─────────────┘     └─────────────┘
```

### Data Flow Summary

```
User Input → Parse → Plan → Check Resources → Get Consent
                                                    │
                                                    ▼
                                            Execute on Backend(s)
                                                    │
                                                    ▼
                                            Collect Results
                                                    │
                                                    ▼
                                            Generate Insights
                                                    │
                                                    ▼
                                            Export/Display
```

---

## Testing Strategy

### Test Coverage Goals

| Component    | Coverage Target |
| ------------ | --------------- |
| Core         | 90%             |
| Backends     | 85%             |
| Intelligence | 80%             |
| Resources    | 85%             |
| CLI          | 75%             |
| Data         | 85%             |

### Mock Strategies

| Component        | Mock Approach                         |
| ---------------- | ------------------------------------- |
| Quantum backends | Mock adapter returning canned results |
| LLM providers    | Mock HTTP responses                   |
| System resources | Inject fake psutil values             |
| File system      | Use tmpdir fixture                    |

---

## Deployment Considerations

### Environment Requirements

| Requirement | Minimum               | Recommended |
| ----------- | --------------------- | ----------- |
| Python      | 3.11                  | 3.12        |
| RAM         | 4 GB                  | 16 GB       |
| Disk        | 500 MB                | 2 GB        |
| OS          | Linux, macOS, Windows | Any         |

### Configuration for Production

```yaml
# Production config recommendations
verbosity: info
llm:
  require_consent: true
  provider: local_preferred # Prefer local for privacy
resources:
  memory_warn_threshold_mb: 1024
  max_execution_time_seconds: 3600
consent:
  remember_decisions: false # Always ask in production
```

---

## 4. Phase Summaries & Usage Guidance

### 4.1 Phase Summary Table

| Phase                     | Duration | Weeks | Key Deliverables                                                             | Mandatory Features       |
| ------------------------- | -------- | ----- | ---------------------------------------------------------------------------- | ------------------------ |
| **Phase 1: Foundation**   | 4 weeks  | 1-4   | CLI skeleton, config system, state machine, logging, basic timer             | 1 (partial), 4 (partial) |
| **Phase 2: Backends**     | 5 weeks  | 5-9   | LRET, Cirq (DM/SV), Qiskit Aer (DM/SV) adapters, registry, plugin system     | 2                        |
| **Phase 3: Intelligence** | 5 weeks  | 10-14 | LLM router (local/remote), auto-selector, insight engine                     | 5, 8                     |
| **Phase 4: Safety**       | 4 weeks  | 15-18 | Resource monitor, consent manager, full execution control                    | 3, 4 (complete)          |
| **Phase 5: Advanced**     | 5 weeks  | 19-23 | Multi-backend comparison, proxima_agent.md, export engine, planning pipeline | 6, 7, 9, 10              |
| **Phase 6: Production**   | 4 weeks  | 24-27 | TUI, comprehensive testing, documentation, packaging                         | 11                       |

**Total: 27 weeks**

---

### 4.2 Phase 1 Summary: Foundation

**What You'll Have:**

- Functional CLI with `proxima run`, `proxima config`, `proxima backends` commands
- Configuration system with hierarchy (CLI > ENV > user config > defaults)
- State machine: IDLE → PLANNING → RUNNING → COMPLETED/ERROR
- Execution timer displaying: task name, elapsed time, current stage
- Basic execution control: Start, Abort

**Usage Example:**

```bash
# Basic invocation
proxima run --backend lret "simulate entanglement"

# With verbose output
proxima run --verbose --backend cirq "quantum teleportation"

# Configuration management
proxima config show
proxima config set verbosity debug
```

---

### 4.3 Phase 2 Summary: Backends

**What You'll Have:**

- Three working backend adapters: LRET, Cirq, Qiskit Aer
- Cirq and Qiskit support both DensityMatrix and StateVector simulators
- Backend registry for plugin-based extensibility
- Backend discovery and capability enumeration

**Usage Example:**

```bash
# List available backends
proxima backends list

# Run with specific backend and simulator type
proxima run --backend cirq --simulator density-matrix "decoherence simulation"
proxima run --backend qiskit --simulator state-vector "grover search"

# Check backend capabilities
proxima backends info lret
```

---

### 4.4 Phase 3 Summary: Intelligence

**What You'll Have:**

- LLM router with support for OpenAI, Anthropic, Ollama, LM Studio
- Automatic backend selection with explanation of reasoning
- Insight engine producing human-readable, analytical interpretations
- Export of results to CSV/XLSX with insights (not raw data)

**Usage Example:**

```bash
# Auto-select backend (agent explains why)
proxima run "simulate decoherence in noisy channel"
# Output: "Selected Cirq with DensityMatrix because decoherence modeling requires..."

# Use local LLM for interpretation
proxima run --llm ollama "quantum error correction"

# Export results with insights
proxima run --export results.xlsx "compare bell states"
```

---

### 4.5 Phase 4 Summary: Safety

**What You'll Have:**

- Resource monitoring: RAM, CPU, low memory detection
- Consent manager: explicit user consent before execution
- Force execute option to bypass warnings
- Full execution control: Start, Abort, Rollback, Pause, Resume

**Usage Example:**

```bash
# Resource-aware execution
proxima run "large-scale simulation"
# Output: "⚠️ This simulation requires ~8GB RAM. Current available: 4GB. Proceed? [y/N/force]"

# Force execution despite warnings
proxima run --force "large-scale simulation"

# Execution control during run
# Press Ctrl+P to pause, Ctrl+R to resume, Ctrl+C to abort
```

---

### 4.6 Phase 5 Summary: Advanced

**What You'll Have:**

- Multi-backend comparison with identical parameters
- proxima_agent.md file support for batch operations
- Planning, Analysis, Execution pipeline
- Session persistence and undo functionality

**Usage Example:**

```bash
# Multi-backend comparison
proxima compare --backends lret,cirq,qiskit "bell state preparation"
# Output: Comparison table with execution times, fidelities, resource usage

# Using proxima_agent.md
proxima run --agent ./proxima_agent.md

# Planning mode (dry-run)
proxima plan "complex multi-step simulation"
```

**proxima_agent.md Example:**

```markdown
# Proxima Agent Instructions

## Goal

Compare quantum entanglement across all backends

## Parameters

- qubits: 4
- shots: 1000
- noise_model: depolarizing

## Steps

1. Run on LRET
2. Run on Cirq (DensityMatrix)
3. Run on Qiskit Aer (StateVector)
4. Compare and generate report

## Output

- Format: XLSX
- Include: fidelity, execution time, resource usage
```

---

### 4.7 Phase 6 Summary: Production

**What You'll Have:**

- Full TUI with Textual (interactive, keyboard-driven)
- Comprehensive test suite (unit, integration, e2e)
- Complete documentation (user guide, developer guide, API reference)
- Multiple distribution channels (PyPI, Homebrew, Docker, binaries)

**Usage Example:**

```bash
# Launch TUI
proxima tui

# Install from PyPI
pip install proxima-agent

# Docker usage
docker run -it proxima-agent run "bell state"
```

---

### 4.8 Feature Verification Checklist

After completing all phases, verify each mandatory feature:

| #   | Feature                  | Verification Command                    | Expected Behavior                                       |
| --- | ------------------------ | --------------------------------------- | ------------------------------------------------------- |
| 1   | Execution Timer          | Run any simulation                      | See task name, elapsed time, current stage in real-time |
| 2   | Backend Selection        | `proxima run "task"` (no --backend)     | Auto-selects and explains reasoning                     |
| 3   | Fail-Safe & Consent      | Run memory-heavy task                   | Warns about resources, asks for consent                 |
| 4   | Execution Control        | Run long task, press Ctrl+P             | Task pauses; Ctrl+R resumes                             |
| 5   | Result Interpretation    | `proxima run --interpret "task"`        | Receives human-readable, analytical insights            |
| 6   | Multi-Backend Comparison | `proxima compare --backends all "task"` | Side-by-side comparison with identical params           |
| 7   | Planning Pipeline        | `proxima plan "complex task"`           | Shows execution plan before running                     |
| 8   | LLM Integration          | `proxima run --llm ollama "task"`       | Uses local LLM; asks consent for remote                 |
| 9   | agent.md                 | `proxima run --agent proxima_agent.md`  | Executes instructions from file                         |
| 10  | Inspired Features        | Various                                 | Plugin system, session persistence, undo                |
| 11  | UI                       | `proxima tui`                           | Full TUI launches                                       |

---

### 4.9 Troubleshooting Guide

| Issue               | Possible Cause                   | Solution                                                     |
| ------------------- | -------------------------------- | ------------------------------------------------------------ |
| Backend not found   | Not installed or not in registry | `proxima backends list` to check; install missing dependency |
| LLM timeout         | Network issues or slow inference | Try local LLM or increase timeout in config                  |
| Out of memory       | Simulation too large             | Reduce qubits/shots, or use --force with caution             |
| Permission denied   | Keyring access issue             | Check OS keychain settings                                   |
| Import error        | Missing dependency               | `pip install proxima-agent[all]` for full installation       |
| State machine stuck | Unhandled exception              | Check logs in `~/.proxima/logs/`, restart session            |

---

### 4.10 Configuration Quick Reference

```yaml
# ~/.proxima/config.yaml

# Verbosity: debug | info | warning | error
verbosity: info

# Output format: text | json | rich
output_format: rich

# Default backend (optional - enables auto-selection if not set)
default_backend: null

# LLM settings
llm:
  provider: ollama # openai | anthropic | ollama | lmstudio
  model: llama3 # Model name for chosen provider
  require_consent: true

# Resource limits
resources:
  memory_warn_threshold_mb: 2048
  max_execution_time_seconds: 1800

# Consent settings
consent:
  remember_decisions: false
  default_action: ask # ask | allow | deny
```

---

### 4.11 Next Steps After Implementation

1. **Extend Backends:** Add support for IBM Quantum, Amazon Braket, IonQ
2. **Web Interface:** Build REST API and React/Vue frontend
3. **Cloud Deployment:** Kubernetes manifests for scalable deployment
4. **ML Integration:** Add machine learning for parameter optimization
5. **Community Plugins:** Enable community-contributed backends and extensions

---

## Appendix: Inspiration Acknowledgments

### From OpenCode AI (NOT COPYING)

- Terminal-centric workflow philosophy
- Session persistence patterns
- Intelligent code understanding approaches

### From Crush (Charmbracelet) (NOT COPYING)

- Beautiful TUI design principles (Bubble Tea/Lipgloss aesthetics)
- User experience considerations
- CLI ergonomics and keyboard-first interaction

**Note:** Proxima is an ORIGINAL creation that takes INSPIRATION from these projects. We do not copy code, architecture, or implementation details. We observe what makes these projects excellent and create our own solutions embodying similar values.

---

_Document Version 2.0 — Prepared for Proxima AI Agent project implementation._
_This HLD follows the required structure: Strategic System Sketch → Phased Roadmap → Phase-by-Phase Implementation Guide → Phase Summaries & Usage Guidance._
