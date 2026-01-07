# How to Build Proxima: A Strategic Guide to Designing an AI Agent for Quantum Simulations

> **Document Version:** 1.0  
> **Last Updated:** January 7, 2026  
> **Status:** High-Level Design Document

---

## Table of Contents

1. [Introduction](#introduction)
2. [Strategic System Sketch](#strategic-system-sketch)
3. [Phased Roadmap](#phased-roadmap)
4. [Phase-by-Phase Implementation Guide](#phase-by-phase-implementation-guide)
5. [Phase Summaries & Usage Guidance](#phase-summaries--usage-guidance)

---

## Introduction

### What is Proxima?

Proxima is an intelligent AI agent designed to orchestrate quantum simulations across multiple backends. It provides a unified interface for selecting, executing, comparing, and interpreting results from various quantum computing frameworks.

### Design Philosophy

Proxima draws architectural and UX inspiration from:

- **OpenCode AI** ([GitHub](https://github.com/opencode-ai/opencode)): For its intelligent code assistance patterns and agent-driven workflows
- **Crush by Charmbracelet** ([GitHub](https://github.com/charmbracelet/crush)): For its elegant terminal UI paradigms and user experience design

However, Proxima is built as a completely independent, extensible system with its own identity.

### Supported Quantum Backends

| Backend        | Simulator Types                         | Repository                                                                             |
| -------------- | --------------------------------------- | -------------------------------------------------------------------------------------- |
| **LRET**       | Framework Integration                   | [kunal5556/LRET](https://github.com/kunal5556/LRET/tree/feature/framework-integration) |
| **Cirq**       | Density Matrix, State Vector            | [quantumlib/Cirq](https://github.com/quantumlib/Cirq)                                  |
| **Qiskit Aer** | Density Matrix, State Vector            | [Qiskit/qiskit-aer](https://github.com/Qiskit/qiskit-aer)                              |
| **Extensible** | Custom backends via plugin architecture | —                                                                                      |

---

## Strategic System Sketch

### Overall Architecture Overview

Proxima follows a **layered modular architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                        │
│         (CLI / Future TUI via Bubble Tea / Future Web UI)           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        AGENT ORCHESTRATION LAYER                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │   Planner   │  │   Executor   │  │   State Machine Manager    │  │
│  └─────────────┘  └──────────────┘  └────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              proxima_agent.md Interpreter                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         INTELLIGENCE LAYER                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  LLM Router      │  │  Backend Selector │  │  Insight Engine  │   │
│  │  (Local/Remote)  │  │  (Auto-Selection) │  │  (Analysis)      │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RESOURCE MANAGEMENT LAYER                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  Memory Monitor  │  │  Execution Timer │  │  Consent Manager │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Execution Control (Start/Abort/Pause/Resume)    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       BACKEND ABSTRACTION LAYER                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
│  │    LRET    │  │    Cirq    │  │ Qiskit Aer │  │  Custom    │     │
│  │  Adapter   │  │  Adapter   │  │  Adapter   │  │  Adapter   │     │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA & OUTPUT LAYER                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  Result Store    │  │  Comparison      │  │  Export Engine   │   │
│  │  (JSON/SQLite)   │  │  Aggregator      │  │  (CSV/XLSX)      │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Core Components Detailed

#### 1. User Interface Layer

**Purpose:** Provides the interaction surface for users to communicate with Proxima.

**Components:**

- **CLI Interface:** Primary interface using libraries like Cobra (Go) or Click/Typer (Python)
- **TUI Interface (Future):** Rich terminal UI using Bubble Tea (Go) or Rich/Textual (Python)
- **Web UI (Future):** React or Svelte-based dashboard

**Data Flow:**

- User commands enter here
- Execution status, timers, and progress displayed here
- All consent prompts surface through this layer

---

#### 2. Agent Orchestration Layer

**Purpose:** The brain of Proxima—plans, coordinates, and manages task execution.

**Components:**

| Component                        | Responsibility                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| **Planner**                      | Decomposes user requests into executable stages; creates task DAGs                   |
| **Executor**                     | Runs tasks according to plan; manages parallel/sequential execution                  |
| **State Machine Manager**        | Tracks execution states (IDLE, PLANNING, RUNNING, PAUSED, ABORTED, COMPLETED, ERROR) |
| **proxima_agent.md Interpreter** | Parses and executes instructions from agent definition files                         |

**Key Behaviors:**

- Explicit planning before execution
- Transparent stage transitions
- Traceable execution history
- Support for future agent.md-based automation

---

#### 3. Intelligence Layer

**Purpose:** Provides smart decision-making capabilities.

**Components:**

| Component            | Function                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------------- |
| **LLM Router**       | Routes requests to appropriate LLM (local vs. remote); manages API keys; enforces consent |
| **Backend Selector** | Analyzes workload characteristics and recommends optimal backend                          |
| **Insight Engine**   | Transforms raw simulation data into human-readable analytical insights                    |

**LLM Router Decision Tree:**

```
User Request → Is LLM needed?
                    │
            ┌───────┴───────┐
            ▼               ▼
           Yes              No
            │                │
    Local LLM available?    Proceed
            │
    ┌───────┴───────┐
    ▼               ▼
   Yes              No
    │                │
User consent?    Use Remote API
    │            (with consent)
    ▼
Use Local LLM
```

---

#### 4. Resource Management Layer

**Purpose:** Ensures safe, transparent, and controllable execution.

**Components:**

- **Memory Monitor:** Continuously tracks RAM usage using psutil or similar; triggers warnings at configurable thresholds
- **Execution Timer:** Tracks wall-clock time per stage and total execution; displays elapsed time in real-time
- **Consent Manager:** Gates all potentially risky or resource-intensive operations behind explicit user approval
- **Execution Controller:** Implements Start, Abort, Pause, Resume operations with proper state persistence

**Fail-Safe Decision Flow:**

```
Before Execution:
  │
  ├─→ Check available memory
  │     └─→ Below threshold? → Warn user → Require consent
  │
  ├─→ Check hardware compatibility
  │     └─→ Incompatible? → Explain risks → Offer "force execute"
  │
  └─→ All checks pass → Proceed with execution
```

---

#### 5. Backend Abstraction Layer

**Purpose:** Provides a unified interface to diverse quantum simulation backends.

**Design Pattern:** Adapter Pattern with Plugin Architecture

**Each Adapter Must Implement:**

- `initialize()` — Set up the backend
- `validate_circuit(circuit)` — Check circuit compatibility
- `execute(circuit, options)` — Run the simulation
- `get_capabilities()` — Report supported features (noise models, qubit limits, etc.)
- `get_resource_requirements()` — Estimate memory/compute needs

**Backend Selection Intelligence:**

```
User Query Analysis:
  │
  ├─→ Extract circuit size (qubit count)
  ├─→ Identify noise requirements
  ├─→ Detect density matrix vs. state vector needs
  ├─→ Check hardware constraints
  │
  └─→ Score each backend → Select highest score → Explain selection
```

---

#### 6. Data & Output Layer

**Purpose:** Manages simulation results, comparisons, and exports.

**Components:**

- **Result Store:** Persists simulation outcomes in structured format (JSON or SQLite)
- **Comparison Aggregator:** Aligns results from multiple backends for side-by-side analysis
- **Export Engine:** Generates CSV, XLSX, and formatted reports

---

### Control & Data Flow Summary

```
┌──────────────────────────────────────────────────────────────────────┐
│                          CONTROL FLOW                                │
│                                                                      │
│  User Command → Planning → Resource Check → Consent → Execute →      │
│  Monitor → Complete/Abort/Pause → Report Results                     │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                  │
│                                                                      │
│  User Input → Parsed Request → Execution Plan → Backend Calls →      │
│  Raw Results → Insight Processing → Formatted Output → User          │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Feature Interconnection Map

| Feature                  | Depends On                           | Feeds Into                       |
| ------------------------ | ------------------------------------ | -------------------------------- |
| Execution Timer          | State Machine                        | UI Layer, Logs                   |
| Backend Selection        | Intelligence Layer, Backend Adapters | Executor                         |
| Fail-Safe                | Memory Monitor, Consent Manager      | Executor (gates execution)       |
| Execution Control        | State Machine                        | All execution paths              |
| Result Interpretation    | Insight Engine, LLM Router           | Export Engine, UI                |
| Multi-Backend Comparison | All Adapters, Comparison Aggregator  | Insight Engine                   |
| LLM Integration          | LLM Router, Consent Manager          | Insight Engine, Backend Selector |
| agent.md Compatibility   | Interpreter, Planner                 | Full orchestration               |

---

## Phased Roadmap

### Overview

The development of Proxima is divided into **six phases**, each building upon the previous to create a fully functional quantum simulation agent.

```
Phase 1: Foundation & Core Infrastructure
    │
    ▼
Phase 2: Backend Integration & Abstraction
    │
    ▼
Phase 3: Intelligence & Decision Systems
    │
    ▼
Phase 4: Safety, Control & Transparency
    │
    ▼
Phase 5: Advanced Features & agent.md Support
    │
    ▼
Phase 6: UI Enhancement & Production Hardening
```

---

### Phase Summary Table

| Phase | Name                 | Duration  | Key Deliverables                                                  |
| ----- | -------------------- | --------- | ----------------------------------------------------------------- |
| 1     | Foundation           | 3-4 weeks | Project structure, CLI scaffold, configuration system             |
| 2     | Backend Integration  | 4-5 weeks | LRET, Cirq, Qiskit adapters; unified interface                    |
| 3     | Intelligence Systems | 4-5 weeks | LLM integration, backend auto-selection, insight engine           |
| 4     | Safety & Control     | 3-4 weeks | Memory monitoring, execution control, consent system              |
| 5     | Advanced Features    | 4-5 weeks | Multi-backend comparison, agent.md interpreter, pipeline planning |
| 6     | UI & Production      | 3-4 weeks | TUI, documentation, testing, deployment                           |

**Total Estimated Duration:** 21-27 weeks

---

## Phase-by-Phase Implementation Guide

---

### Phase 1: Foundation & Core Infrastructure

**Objective:** Establish the project skeleton, development environment, and foundational systems.

---

#### Step 1.1: Project Initialization

**Language Selection Decision:**

- **Option A (Recommended for Performance):** Go
  - Use Go modules for dependency management
  - Leverage goroutines for concurrent execution
  - Utilize Cobra for CLI framework
- **Option B (Recommended for Quantum Ecosystem):** Python
  - Use Poetry or PDM for dependency management
  - Leverage asyncio for concurrent execution
  - Utilize Typer or Click for CLI framework

**Directory Structure to Create:**

```
proxima/
├── cmd/                    # CLI entry points
├── internal/               # Private application code
│   ├── agent/              # Orchestration logic
│   ├── backends/           # Backend adapters
│   ├── intelligence/       # LLM and decision systems
│   ├── resources/          # Resource management
│   ├── ui/                 # User interface components
│   └── utils/              # Shared utilities
├── pkg/                    # Public libraries (if needed)
├── configs/                # Configuration files
├── docs/                   # Documentation
├── tests/                  # Test suites
├── proxima_agent.md        # Agent definition file (for Phase 5)
└── README.md
```

---

#### Step 1.2: Configuration System

**Tools to Use:**

- **Go:** Viper for configuration management
- **Python:** Pydantic Settings or Dynaconf

**Configuration Hierarchy to Implement:**

1. Default values (hardcoded)
2. Configuration file (`~/.proxima/config.yaml`)
3. Environment variables (prefixed with `PROXIMA_`)
4. Command-line flags (highest priority)

**Configuration Categories:**

- General settings (verbosity, output format)
- Backend configurations (API endpoints, credentials)
- LLM settings (API keys, model preferences, local LLM paths)
- Resource thresholds (memory limits, timeout values)
- Consent preferences (auto-approve certain actions)

---

#### Step 1.3: Logging & Telemetry Foundation

**Logging Framework:**

- **Go:** Zap or Zerolog for structured logging
- **Python:** Structlog or Loguru

**Log Levels to Implement:**

- DEBUG: Detailed execution information
- INFO: General operational messages
- WARN: Resource warnings, consent prompts
- ERROR: Failures and exceptions

**Telemetry Considerations:**

- Execution timing metrics
- Backend usage statistics
- Error frequency tracking
- All telemetry must be opt-in with explicit consent

---

#### Step 1.4: CLI Scaffold

**Commands to Implement:**

| Command            | Description                        |
| ------------------ | ---------------------------------- |
| `proxima init`     | Initialize configuration           |
| `proxima config`   | View/modify settings               |
| `proxima run`      | Execute a simulation (placeholder) |
| `proxima backends` | List available backends            |
| `proxima version`  | Display version information        |

**CLI Flags to Support:**

- `--verbose` / `-v`: Increase output verbosity
- `--config`: Specify configuration file path
- `--backend`: Explicitly select backend
- `--dry-run`: Show plan without executing
- `--force`: Skip consent prompts (with warnings)

---

#### Step 1.5: State Machine Foundation

**States to Define:**

```
IDLE → PLANNING → READY → RUNNING → COMPLETED
                    │        │
                    │        ├──→ PAUSED → RUNNING (resume)
                    │        │
                    │        └──→ ABORTED
                    │
                    └──→ ERROR
```

**State Transition Rules:**

- Only valid transitions allowed
- Each transition logged with timestamp
- State persisted for recovery

**Implementation Approach:**

- **Go:** Use a finite state machine library or implement custom
- **Python:** Use transitions library or implement custom

---

### Phase 2: Backend Integration & Abstraction

**Objective:** Create adapters for quantum backends with a unified interface.

---

#### Step 2.1: Define Backend Interface Contract

**Abstract Interface Methods:**

| Method                               | Purpose                        |
| ------------------------------------ | ------------------------------ |
| `get_name()`                         | Returns backend identifier     |
| `get_version()`                      | Returns backend version        |
| `get_capabilities()`                 | Returns feature dictionary     |
| `get_resource_requirements(circuit)` | Estimates memory/compute needs |
| `validate(circuit)`                  | Checks circuit compatibility   |
| `execute(circuit, options)`          | Runs simulation                |
| `get_result_schema()`                | Returns expected output format |

**Capability Flags:**

- `supports_density_matrix`: Boolean
- `supports_state_vector`: Boolean
- `supports_noise_models`: Boolean
- `max_qubits`: Integer
- `supports_gpu`: Boolean

---

#### Step 2.2: LRET Adapter Implementation

**Integration Steps:**

1. Add LRET framework-integration branch as dependency
2. Create adapter class implementing the interface
3. Map LRET's native API to unified interface
4. Implement circuit translation layer (if needed)
5. Handle LRET-specific error types
6. Write adapter-specific unit tests

**LRET-Specific Considerations:**

- Framework integration branch features
- Custom simulation modes
- Result format normalization

---

#### Step 2.3: Cirq Adapter Implementation

**Integration Steps:**

1. Add Cirq as dependency via pip/poetry
2. Create adapter class with dual simulator support
3. Implement Density Matrix simulator path
4. Implement State Vector simulator path
5. Add simulator selection logic based on use case
6. Normalize Cirq result objects to common format

**Cirq Simulator Selection Logic:**

- Use Density Matrix for: noise simulation, mixed states, smaller circuits
- Use State Vector for: pure states, larger circuits (memory permitting)

---

#### Step 2.4: Qiskit Aer Adapter Implementation

**Integration Steps:**

1. Add qiskit-aer as dependency
2. Create adapter class with dual simulator support
3. Implement Density Matrix simulator path via AerSimulator
4. Implement State Vector simulator path via AerSimulator
5. Support Qiskit's transpilation pipeline
6. Handle Qiskit-specific job patterns

**Qiskit-Specific Features:**

- Noise model integration
- Backend options configuration
- Shot-based vs. statevector execution

---

#### Step 2.5: Backend Registry & Discovery

**Registry Responsibilities:**

- Maintain list of available backends
- Lazy-load adapters on demand
- Report backend health status
- Support dynamic backend registration

**Discovery Mechanism:**

1. Scan for installed quantum packages
2. Verify each backend is functional
3. Cache capabilities for quick access
4. Re-scan on user request

---

#### Step 2.6: Unified Result Format

**Common Result Schema:**

```
{
  "backend": "string",
  "simulator_type": "density_matrix | state_vector",
  "execution_time_ms": "number",
  "qubit_count": "number",
  "result_type": "counts | statevector | density_matrix",
  "data": { ... },
  "metadata": { ... }
}
```

**Result Normalization:**

- Convert backend-specific formats to common schema
- Preserve original data in metadata
- Standardize probability representations

---

### Phase 3: Intelligence & Decision Systems

**Objective:** Integrate LLM capabilities and intelligent decision-making.

---

#### Step 3.1: LLM Router Architecture

**Components to Build:**

1. **Provider Registry:** Tracks available LLM providers
2. **Local LLM Detector:** Finds locally installed models
3. **API Key Manager:** Securely stores and retrieves keys
4. **Request Router:** Directs queries to appropriate provider
5. **Consent Gate:** Enforces user approval before LLM calls

**Supported Provider Types:**

| Type              | Examples                     | Detection Method                |
| ----------------- | ---------------------------- | ------------------------------- |
| Remote API        | OpenAI, Anthropic, Google    | API key presence                |
| Local Inference   | Ollama, LM Studio, llama.cpp | Process detection, socket check |
| Local Model Files | GGUF, GGML files             | File system scan                |

---

#### Step 3.2: Local LLM Integration

**Detection Strategies:**

1. Check for running Ollama service (default port 11434)
2. Check for LM Studio server (configurable port)
3. Scan configured directories for model files
4. Verify model compatibility and readiness

**Local LLM Interface:**

- Use OpenAI-compatible API format when available
- Fall back to native API for specific runtimes
- Support model selection from available local models

**User Distinction Requirements:**

- Clear labeling: "[LOCAL LLM]" vs "[REMOTE API]"
- Separate consent prompts for each type
- Log which provider handled each request

---

#### Step 3.3: Remote API Integration

**Providers to Support:**

1. **OpenAI:** GPT-4, GPT-4-turbo models
2. **Anthropic:** Claude models
3. **Google:** Gemini models
4. **Custom:** User-defined OpenAI-compatible endpoints

**API Key Management:**

- Store encrypted in system keychain (keyring library)
- Support environment variable fallback
- Never log or display API keys
- Validate keys on startup (optional)

---

#### Step 3.4: Consent Management System

**Consent Types:**

| Action                        | Consent Level                      |
| ----------------------------- | ---------------------------------- |
| Use local LLM                 | Explicit per-session or persistent |
| Use remote API                | Explicit per-session or persistent |
| Modify backend logic          | Always explicit                    |
| Force execute (low resources) | Always explicit                    |
| Execute untrusted agent.md    | Always explicit                    |

**Consent Storage:**

- Session consents: In-memory only
- Persistent consents: Encrypted configuration file
- Audit log: Record all consent decisions

**Consent Prompt Format:**

```
╭─────────────────────────────────────────────────────╮
│ CONSENT REQUIRED                                    │
├─────────────────────────────────────────────────────┤
│ Action: Use remote LLM (OpenAI GPT-4)               │
│ Reason: Analyze simulation results                  │
│ Data sent: Summary statistics (no raw data)         │
├─────────────────────────────────────────────────────┤
│ [Y] Approve this time                               │
│ [A] Always approve this action                      │
│ [N] Deny                                            │
╰─────────────────────────────────────────────────────╯
```

---

#### Step 3.5: Backend Auto-Selection Intelligence

**Selection Algorithm:**

1. **Parse User Query:** Extract circuit characteristics
2. **Analyze Requirements:**
   - Qubit count
   - Gate types used
   - Noise requirements
   - Output type needed (counts vs. statevector)
3. **Score Backends:** Rate each backend on:
   - Feature compatibility (required features supported)
   - Performance (historical execution times)
   - Resource fit (memory requirements vs. available)
4. **Select Best:** Choose highest-scoring backend
5. **Explain Selection:** Generate human-readable justification

**Explanation Template:**

```
Selected backend: Qiskit Aer (State Vector Simulator)
Reason: Your circuit has 12 qubits with no noise requirements.
        State vector simulation provides exact amplitudes.
        Estimated memory: 128 MB (available: 8 GB)
        Estimated time: ~2 seconds
```

---

#### Step 3.6: Insight Engine

**Purpose:** Transform raw simulation data into actionable insights.

**Analysis Capabilities:**

1. **Statistical Analysis:**

   - Probability distributions
   - Entropy calculations
   - Fidelity metrics

2. **Comparative Analysis:**

   - Backend result differences
   - Performance comparisons
   - Accuracy assessments

3. **Visualization Recommendations:**
   - Suggest appropriate chart types
   - Highlight significant patterns
   - Flag anomalies

**LLM-Assisted Interpretation:**

- Feed summarized results to LLM
- Request natural language explanations
- Generate decision recommendations
- Always with user consent

---

### Phase 4: Safety, Control & Transparency

**Objective:** Implement resource monitoring, execution control, and transparency features.

---

#### Step 4.1: Memory Monitoring System

**Implementation Using:**

- **Python:** psutil library
- **Go:** gopsutil library

**Monitoring Metrics:**

| Metric         | Description                | Threshold                     |
| -------------- | -------------------------- | ----------------------------- |
| Available RAM  | Free memory for allocation | Configurable (default: 500MB) |
| Process Memory | Proxima's own usage        | Configurable (default: 2GB)   |
| Swap Usage     | Virtual memory pressure    | Warning at 50%                |

**Monitoring Workflow:**

1. Check resources before execution starts
2. Continuous monitoring during execution
3. Trigger warnings at configurable thresholds
4. Pause execution if critical levels reached
5. Resume only with user consent

---

#### Step 4.2: Execution Timer & Progress Tracking

**Timer Components:**

1. **Global Timer:** Total execution time
2. **Stage Timer:** Per-stage elapsed time
3. **ETA Calculator:** Estimated time remaining

**Display Format:**

```
╭─ Execution Status ──────────────────────────────────╮
│ Overall: ██████████░░░░░░░░░░ 50% │ 2m 34s elapsed  │
├─────────────────────────────────────────────────────┤
│ ✓ Planning          │ 0.3s                          │
│ ✓ Backend Init      │ 1.2s                          │
│ → Executing Circuit │ 2m 32s (running...)           │
│ ○ Result Analysis   │ pending                       │
│ ○ Insight Generation│ pending                       │
╰─────────────────────────────────────────────────────╯
```

**Progress Events:**

- Stage start/complete
- Percentage updates (for long operations)
- Warning events
- Error events

---

#### Step 4.3: Execution Control Implementation

**Control Operations:**

| Operation  | Implementation                                       |
| ---------- | ---------------------------------------------------- |
| **Start**  | Initialize resources, begin execution pipeline       |
| **Abort**  | Immediate termination, cleanup resources, log reason |
| **Pause**  | Suspend at next safe checkpoint, preserve state      |
| **Resume** | Restore state, continue from checkpoint              |

**Pause/Resume Mechanism:**

1. Define safe checkpoint locations in execution flow
2. At checkpoints, check for pause signal
3. If paused, serialize current state to disk
4. On resume, deserialize and continue

**Abort Cleanup Checklist:**

- Release backend resources
- Close file handles
- Save partial results (if any)
- Log abort reason and state
- Return to IDLE state

---

#### Step 4.4: State Visibility & Traceability

**State Transition Logging:**

```
[2026-01-07 10:23:45] State: IDLE → PLANNING
[2026-01-07 10:23:45] State: PLANNING → READY
[2026-01-07 10:23:46] State: READY → RUNNING
[2026-01-07 10:25:12] State: RUNNING → PAUSED (user request)
[2026-01-07 10:26:30] State: PAUSED → RUNNING (resumed)
[2026-01-07 10:28:45] State: RUNNING → COMPLETED
```

**Execution History:**

- Persist last N executions
- Include: start time, duration, backend, status, result summary
- Queryable via CLI command

---

#### Step 4.5: Hardware Compatibility Checks

**Checks to Perform:**

1. **GPU Availability:** For GPU-accelerated backends
2. **CUDA Version:** If GPU backend selected
3. **CPU Features:** AVX support for certain optimizations
4. **Memory Architecture:** 32-bit vs 64-bit limitations

**Incompatibility Handling:**

```
╭─ Hardware Warning ──────────────────────────────────╮
│ The selected backend (Qiskit Aer GPU) requires      │
│ CUDA 11.0+, but CUDA 10.2 was detected.             │
├─────────────────────────────────────────────────────┤
│ Risks of forcing execution:                         │
│ • Simulation may fail partway through               │
│ • Results may be incorrect                          │
│ • System instability possible                       │
├─────────────────────────────────────────────────────┤
│ [F] Force execute anyway (not recommended)          │
│ [S] Switch to CPU backend (recommended)             │
│ [C] Cancel                                          │
╰─────────────────────────────────────────────────────╯
```

---

### Phase 5: Advanced Features & agent.md Support

**Objective:** Implement multi-backend comparison, pipeline planning, and agent.md support.

---

#### Step 5.1: Multi-Backend Comparison Framework

**Comparison Workflow:**

1. User specifies multiple backends or selects "compare all"
2. Proxima validates circuit compatibility with each backend
3. Execute same simulation on each backend (parallel if possible)
4. Collect and normalize results
5. Generate comparative analysis

**Parallel Execution Strategy:**

- Use process pools for CPU-bound backends
- Respect memory constraints (may need sequential execution)
- Track individual backend timings

**Comparison Report Structure:**

```
╭─ Multi-Backend Comparison Report ───────────────────╮
│ Circuit: 8-qubit Grover's Algorithm                 │
│ Backends Compared: 3                                │
├─────────────────────────────────────────────────────┤
│ Backend          │ Time    │ Memory  │ Status       │
├──────────────────┼─────────┼─────────┼──────────────┤
│ LRET             │ 1.23s   │ 256 MB  │ ✓ Success    │
│ Cirq (SV)        │ 0.89s   │ 128 MB  │ ✓ Success    │
│ Qiskit Aer (SV)  │ 0.95s   │ 130 MB  │ ✓ Success    │
├─────────────────────────────────────────────────────┤
│ Result Consistency: 99.97% agreement                │
│ Fastest: Cirq (State Vector)                        │
│ Most Memory Efficient: Cirq (State Vector)          │
╰─────────────────────────────────────────────────────╯
```

---

#### Step 5.2: Pipeline Planning System

**Planning Stages:**

1. **Request Parsing:** Understand what user wants
2. **Requirement Analysis:** Determine needed resources and backends
3. **Dependency Resolution:** Order tasks by dependencies
4. **Resource Allocation:** Assign backends and compute resources
5. **Execution Plan Generation:** Create detailed step-by-step plan

**Plan Representation (DAG):**

```
Parse Input → Validate Circuit → Select Backend(s)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              Execute on         Execute on          Execute on
                LRET              Cirq               Qiskit Aer
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        ▼
                              Aggregate Results
                                        │
                                        ▼
                              Generate Insights
                                        │
                                        ▼
                              Export Report
```

**Plan Display Before Execution:**

```
╭─ Execution Plan ────────────────────────────────────╮
│ The following steps will be executed:               │
├─────────────────────────────────────────────────────┤
│ 1. Parse and validate input circuit                 │
│ 2. Initialize backends: LRET, Cirq                  │
│ 3. Execute simulation on LRET                       │
│ 4. Execute simulation on Cirq (parallel)            │
│ 5. Compare and aggregate results                    │
│ 6. Generate insight report                          │
│ 7. Export to CSV                                    │
├─────────────────────────────────────────────────────┤
│ Estimated total time: 3-5 minutes                   │
│ Estimated memory peak: 512 MB                       │
├─────────────────────────────────────────────────────┤
│ [P] Proceed   [M] Modify plan   [C] Cancel          │
╰─────────────────────────────────────────────────────╯
```

---

#### Step 5.3: proxima_agent.md Interpreter

**File Format Specification:**

```markdown
# proxima_agent.md

## Metadata

- version: 1.0
- author: user
- created: 2026-01-07

## Configuration

- default_backend: auto
- parallel_execution: true
- insight_level: detailed

## Tasks

### Task: Run Quantum Simulation

- circuit_file: grover_8qubit.qasm
- backends: [cirq, qiskit_aer]
- compare_results: true
- export_format: xlsx

### Task: Analyze Results

- use_llm: local_preferred
- analysis_type: statistical
- generate_recommendations: true
```

**Interpreter Workflow:**

1. Parse markdown file using markdown parser
2. Extract metadata and configuration
3. Build task list from Task sections
4. Validate each task against capabilities
5. Generate execution plan
6. Request user consent for sensitive operations
7. Execute plan

**Security Considerations:**

- Validate file integrity before parsing
- Sandbox any file path references
- Require explicit consent for LLM usage
- Log all interpreted actions

---

#### Step 5.4: Result Export & Interpretation

**Export Formats:**

| Format | Use Case                         | Library                 |
| ------ | -------------------------------- | ----------------------- |
| CSV    | Data analysis, spreadsheets      | csv (stdlib)            |
| XLSX   | Rich formatting, multiple sheets | openpyxl                |
| JSON   | Programmatic access              | json (stdlib)           |
| PDF    | Reports (future)                 | reportlab or weasyprint |

**Insight Generation Pipeline:**

1. Load raw results
2. Apply statistical analysis
3. Identify patterns and anomalies
4. Generate natural language summary (with LLM if consented)
5. Create visualizations (matplotlib, plotly)
6. Compile into structured report

**Human-Readable Insight Example:**

```
╭─ Simulation Insights ───────────────────────────────╮
│                                                     │
│ Summary:                                            │
│ The Grover's algorithm simulation successfully      │
│ amplified the target state |101⟩ to 97.3%           │
│ probability after 2 iterations, consistent with     │
│ theoretical predictions.                            │
│                                                     │
│ Key Findings:                                       │
│ • Target state probability: 97.3%                   │
│ • Theoretical optimum: 97.8%                        │
│ • Fidelity: 99.5%                                   │
│ • All non-target states suppressed below 1%         │
│                                                     │
│ Recommendations:                                    │
│ • Results are valid for algorithm verification      │
│ • Consider noise simulation for hardware estimates  │
│                                                     │
╰─────────────────────────────────────────────────────╯
```

---

#### Step 5.5: Additional Features (Inspired by OpenCode AI & Crush)

**Feature: Multi-Model Support**

- **Value:** Users can leverage different LLMs for different tasks
- **Implementation:** Model router with task-to-model mapping
- **Use Case:** Use fast model for quick analysis, powerful model for complex interpretation

**Feature: Flexible LLM Switching**

- **Value:** Switch between local and remote LLMs based on task sensitivity
- **Implementation:** Runtime model switching via command or config
- **Use Case:** Use local for sensitive data, remote for general queries

**Feature: Session Persistence**

- **Value:** Resume previous work after restart
- **Implementation:** Serialize session state to disk
- **Use Case:** Long-running simulations, computer restarts

**Feature: Plugin System**

- **Value:** Extend Proxima without modifying core
- **Implementation:** Plugin discovery and loading mechanism
- **Use Case:** Custom backends, custom analysis modules

**Feature: Batch Execution**

- **Value:** Run multiple simulations unattended
- **Implementation:** Queue-based execution with notifications
- **Use Case:** Parameter sweeps, overnight runs

**Feature: Result Caching**

- **Value:** Avoid redundant computations
- **Implementation:** Content-addressed cache with configurable TTL
- **Use Case:** Repeated simulations, iterative development

---

### Phase 6: UI Enhancement & Production Hardening

**Objective:** Polish the user experience and prepare for production deployment.

---

#### Step 6.1: Terminal UI (TUI) Implementation

**Framework Options:**

- **Go:** Bubble Tea (bubbletea) with Lip Gloss for styling
- **Python:** Textual or Rich

**TUI Components to Build:**

1. **Dashboard View:** Overview of system status, recent executions
2. **Execution View:** Real-time progress, logs, timers
3. **Configuration View:** Interactive settings management
4. **Results Browser:** Navigate and inspect past results
5. **Backend Manager:** View and configure backends

**Design Principles (Inspired by Crush):**

- Minimal, clean aesthetic
- Responsive to terminal size
- Keyboard-first navigation
- Contextual help available
- Consistent color theming

---

#### Step 6.2: Error Handling & Recovery

**Error Categories:**

| Category       | Handling Strategy                    |
| -------------- | ------------------------------------ |
| User Error     | Clear message, suggest correction    |
| Backend Error  | Retry logic, fallback backend option |
| Resource Error | Pause, warn, wait for consent        |
| System Error   | Log, notify, graceful degradation    |

**Recovery Mechanisms:**

- Checkpoint-based recovery for long operations
- Automatic retry with exponential backoff
- Fallback backend selection on failure
- Partial result preservation

---

#### Step 6.3: Testing Strategy

**Test Levels:**

1. **Unit Tests:** Individual functions and methods
2. **Integration Tests:** Component interactions
3. **Backend Tests:** Adapter functionality with mock backends
4. **End-to-End Tests:** Full workflow scenarios
5. **Performance Tests:** Resource usage, execution time benchmarks

**Testing Frameworks:**

- **Python:** pytest with pytest-asyncio, pytest-mock
- **Go:** testing package with testify

**CI/CD Pipeline:**

- Run tests on every pull request
- Linting and formatting checks
- Security scanning
- Build artifacts for releases

---

#### Step 6.4: Documentation

**Documentation Types:**

1. **User Guide:** How to use Proxima
2. **API Reference:** For programmatic usage
3. **Developer Guide:** How to extend Proxima
4. **Backend Guide:** How to add new backends
5. **agent.md Reference:** File format specification

**Documentation Tools:**

- **Python:** Sphinx or MkDocs
- **Go:** godoc with Hugo or MkDocs

---

#### Step 6.5: Packaging & Distribution

**Distribution Methods:**

1. **PyPI Package** (if Python): `pip install proxima-agent`
2. **Homebrew Tap** (macOS): `brew install proxima`
3. **Binary Releases** (all platforms): GitHub Releases
4. **Container Image**: Docker Hub or GitHub Container Registry

**Packaging Checklist:**

- Version management (semantic versioning)
- Changelog maintenance
- License file inclusion
- Dependency locking

---

## Phase Summaries & Usage Guidance

---

### Phase 1 Summary: Foundation

**Features Implemented:**

- Project structure and development environment
- Configuration system with hierarchical loading
- Basic CLI with core commands
- Logging infrastructure
- State machine foundation

**New Capabilities:**

- Initialize Proxima with custom configuration
- View and modify settings
- Basic command structure ready for extension

**Usage After Phase 1:**

```
# Initialize Proxima
proxima init

# View configuration
proxima config show

# Set a configuration value
proxima config set verbosity debug

# Check version
proxima version
```

---

### Phase 2 Summary: Backend Integration

**Features Implemented:**

- LRET adapter with full simulation support
- Cirq adapter (Density Matrix + State Vector)
- Qiskit Aer adapter (Density Matrix + State Vector)
- Backend registry and discovery
- Unified result format

**New Capabilities:**

- List available backends and their capabilities
- Run simulations on any supported backend
- Receive normalized results regardless of backend

**Usage After Phase 2:**

```
# List available backends
proxima backends list

# Show backend details
proxima backends info cirq

# Run a simulation on specific backend
proxima run --backend qiskit_aer --circuit circuit.qasm

# Run with automatic backend selection (placeholder)
proxima run --circuit circuit.qasm
```

---

### Phase 3 Summary: Intelligence Systems

**Features Implemented:**

- LLM Router with local/remote support
- API key management for remote LLMs
- Local LLM detection and integration
- Backend auto-selection intelligence
- Insight engine for result interpretation
- Consent management system

**New Capabilities:**

- Automatic backend selection with explanation
- LLM-assisted result analysis (with consent)
- Switch between local and remote LLMs
- Human-readable insights from simulation data

**Usage After Phase 3:**

```
# Run with auto-selection (now functional)
proxima run --circuit circuit.qasm
# Output: Selected backend: Cirq (State Vector)
#         Reason: 8-qubit pure state simulation...

# Get insights on results
proxima analyze --results results.json --use-llm local

# Configure LLM settings
proxima config set llm.provider openai
proxima config set llm.local_path /path/to/ollama
```

---

### Phase 4 Summary: Safety & Control

**Features Implemented:**

- Memory monitoring with configurable thresholds
- Execution timer with stage tracking
- Start/Abort/Pause/Resume operations
- State visibility and transition logging
- Hardware compatibility checks
- Force execute with explicit consent

**New Capabilities:**

- Real-time execution progress display
- Pause long-running simulations
- Resume from checkpoints
- Clear warnings before risky operations
- Full execution history

**Usage After Phase 4:**

```
# Run with real-time progress
proxima run --circuit circuit.qasm
# Shows: ██████░░░░ 60% | 1m 23s elapsed

# Pause execution (during run, press Ctrl+P)
# Resume execution
proxima resume --session latest

# Abort execution (during run, press Ctrl+C)
# View execution history
proxima history

# Force execute despite warnings
proxima run --circuit large_circuit.qasm --force
```

---

### Phase 5 Summary: Advanced Features

**Features Implemented:**

- Multi-backend comparison framework
- Pipeline planning with DAG visualization
- proxima_agent.md interpreter
- Enhanced result export (CSV, XLSX)
- Detailed insight generation
- Plugin system foundation
- Session persistence
- Result caching

**New Capabilities:**

- Compare same simulation across multiple backends
- Plan execution before running
- Automate workflows via agent.md files
- Export rich reports
- Resume sessions after restart

**Usage After Phase 5:**

```
# Compare across backends
proxima compare --circuit circuit.qasm --backends cirq,qiskit_aer,lret

# Show execution plan
proxima plan --circuit circuit.qasm --compare-all

# Execute from agent.md file
proxima agent run proxima_agent.md

# Export results
proxima export --session latest --format xlsx --output report.xlsx

# Resume previous session
proxima session resume
```

---

### Phase 6 Summary: UI & Production

**Features Implemented:**

- Full Terminal UI (TUI)
- Comprehensive error handling
- Complete test coverage
- Full documentation
- Multiple distribution packages

**New Capabilities:**

- Interactive dashboard
- Visual execution monitoring
- In-app configuration
- Results browser

**Usage After Phase 6:**

```
# Launch interactive TUI
proxima ui

# TUI provides:
# - Dashboard with system overview
# - Real-time execution monitoring
# - Configuration management
# - Results browsing and analysis
# - Backend management

# Standard CLI remains available
proxima run --circuit circuit.qasm
```

---

## Appendix

### A. Technology Stack Summary

| Component        | Python Option               | Go Option              |
| ---------------- | --------------------------- | ---------------------- |
| CLI Framework    | Typer, Click                | Cobra                  |
| TUI Framework    | Textual, Rich               | Bubble Tea             |
| Configuration    | Pydantic Settings, Dynaconf | Viper                  |
| Logging          | Structlog, Loguru           | Zap, Zerolog           |
| Async            | asyncio                     | goroutines             |
| Testing          | pytest                      | testing + testify      |
| State Machine    | transitions                 | custom or statemachine |
| HTTP Client      | httpx, aiohttp              | net/http               |
| Resource Monitor | psutil                      | gopsutil               |
| Keyring          | keyring                     | go-keyring             |

### B. Quantum Libraries Reference

| Library    | Purpose                      | Documentation         |
| ---------- | ---------------------------- | --------------------- |
| Cirq       | Google's quantum framework   | quantumai.google/cirq |
| Qiskit     | IBM's quantum SDK            | qiskit.org            |
| Qiskit Aer | High-performance simulators  | qiskit.org/aer        |
| LRET       | Custom framework integration | GitHub repository     |

### C. LLM Integration Reference

| Provider  | API Style         | Local Option |
| --------- | ----------------- | ------------ |
| OpenAI    | REST API          | —            |
| Anthropic | REST API          | —            |
| Ollama    | OpenAI-compatible | Yes          |
| LM Studio | OpenAI-compatible | Yes          |
| llama.cpp | Native API        | Yes          |

---

## Final Notes

This document provides a comprehensive blueprint for building Proxima. Each phase builds upon the previous, allowing for incremental development and testing.

**Key Success Factors:**

1. **Modularity:** Keep components loosely coupled
2. **Transparency:** Never hide what Proxima is doing
3. **Consent:** Always ask before sensitive operations
4. **Extensibility:** Design for future backends and features
5. **User Focus:** Prioritize clear communication and usability

**Next Steps:**

1. Review this document thoroughly
2. Set up development environment
3. Begin Phase 1 implementation
4. Iterate based on testing and feedback

---

## Appendix: Documentation Standards

### Part 1: Markdown Formatting Guidelines

**User Instructions Take Precedence:** If the user provides specific instructions about the desired output format, these instructions should always take precedence over the default formatting guidelines outlined below.

**Heading Structure:**

- **Main Title (#):** Use once at the top for the document's primary title
- **Primary Subheadings (##):** Use multiple times for main sections

**Paragraph Guidelines:**

- Keep paragraphs short (3-5 sentences) to avoid dense text blocks
- Ensure logical flow between sections

**List Formatting:**

- Use `-` or `*` for unordered lists
- Use numbers (`1.`, `2.`) for ordered lists
- Combine bullet points or numbered lists for steps, key takeaways, or grouped ideas

**Key Principle:** Ensure headings and lists flow logically, making it easy for readers to scan and understand key points quickly. The readability and format of the output is very important.

---

### Part 2: Citation Guidelines

**Citation Format Preservation:**

- **IMPORTANT:** Preserve any and all citations following the `【{cursor}†L{line_start}(-L{line_end})?】` format

**Image Embedding Rules:**

1. If embedding images with `【{cursor}†embed_image】`, ALWAYS cite them at the **BEGINNING of paragraphs**
2. Do NOT mention the sources of the embed_image citation (they are automatically displayed in the UI)
3. Do NOT use embed_image citations in front of headers
4. ONLY embed images at paragraphs containing three to five sentences minimum

**Image Search Policy:**

- **No proactive image searching:** Do not specifically search for images to embed
- If images are encountered while researching the main issue, they may be considered
- Do not go out of the way to find images to embed
- Lower resolution images are acceptable; no need to seek higher resolution versions

**Image Citation Restrictions:**

- ONLY embed images that have been actually clicked into/opened
- Do NOT cite the same image more than once
- If an unsupported content type error message appears for an image, embedding will NOT work—skip it

---

### Part 3: Comprehensiveness Guidelines

**Core Principle:** Responses MUST be extremely comprehensive. This means providing fully detailed, complete, deep, and accurate explanations for user queries.

**Depth Requirements:**

- Do not settle for surface-level summaries or high-level overviews
- Dive into technical, procedural, contextual, and conceptual depths of the topic
- Responses should leave no significant gaps or unanswered questions

**Content Inclusion:**

- Include background information, definitions, or contextual clarifications when needed to ensure clarity or completeness
- Only omit such information if explicitly told not to include it
- Always lean toward thoroughness over brevity, especially when the user expresses interest in a complete understanding

**Contextual Additions:**

- Relevant background introduction or context may be included
- Ensure any additions do not conflict with user instructions
- Prioritize user-specified requirements over default comprehensiveness when conflicts arise

**Quality Standards:**

1. **Accuracy:** All information must be verified and correct
2. **Completeness:** Cover all relevant aspects of the topic
3. **Clarity:** Ensure explanations are understandable at the appropriate level
4. **Relevance:** Stay focused on the user's query while providing necessary context
5. **Depth:** Go beyond surface-level explanations to provide true understanding

---

### Part 4: Stay Updated Guidelines

**Core Principle:** Internal knowledge may be outdated. DO NOT rely solely on training data or memorized information.

**Research Requirements:**

- Use searches to gather the latest insights before diving deeper into any topic
- Understand the current state of research and developments
- Verify information against up-to-date sources when possible

**Outdated Information Warning Signs:**

- If a user asks for a recent update but the answer only contains facts known before 2024, the response is on the wrong track
- Current year context: It is now 2026—ensure information reflects this timeframe
- Be especially cautious with rapidly evolving fields (technology, frameworks, APIs, regulations)

**Best Practices:**

1. **Verify Currency:** Check if the topic has had recent developments or changes
2. **Search First:** When in doubt, search for recent information before answering
3. **Acknowledge Limitations:** If unable to verify current information, clearly state this
4. **Prioritize Recency:** For time-sensitive topics, recent sources take precedence over older ones
5. **Cross-Reference:** Use multiple sources to confirm current state of information

**Domains Requiring Extra Vigilance:**

- Software versions and API changes
- Library and framework updates
- Security vulnerabilities and patches
- Regulatory and compliance changes
- Market conditions and industry trends
- Research breakthroughs and discoveries

---

### Part 5: Connected Source Citations

**Core Principle:** It is critical to cite connected sources in responses when they have been used. Users need to know where information came from.

**Citation Requirements:**

- **MUST cite actual page opens as sources**
- The cursor format `【{cursor}†L{line_start}(-L{line_end})?】` MUST correspond with a tether_id from a `browse.open` call
- Such `browse.open` call CANNOT be a search result

**Prohibited Citation Types:**

- **CANNOT cite Search results for query (`browse.search`) results as a source**
- Search results will not be properly displayed to the user
- Only cite sources that have been actually opened and read

**Handling Missing Information:**

- If the information the user is asking for is NOT found in the connected sources:
  - State clearly in the report that the information was not found in the connected sources
  - Do not fabricate or assume information that wasn't in the sources

**Error Handling:**

- If errors were encountered while searching over the sources:
  - State that clearly in the report
  - Explain what sources were successfully accessed vs. which failed
  - Provide whatever information was gathered from successful sources

**Best Practices:**

1. **Traceability:** Every factual claim should be traceable to a cited source
2. **Accuracy:** Ensure citations point to the correct line ranges
3. **Transparency:** Be upfront about source limitations or access issues
4. **Completeness:** Cite all sources that contributed to the response
5. **Honesty:** Never cite sources that weren't actually accessed

---

### Part 6: Deep Research Task Constraints

**IMPORTANT: These constraints are non-negotiable for deep research tasks.**

**Code & File Modification Prohibitions:**

- **Do NOT make any changes to any existing code or files**
- No refactoring
- No edits
- No silent assumptions

**Code Implementation Prohibitions:**

- **Do NOT provide any actual code implementations**
- Absolutely no code snippets
- Absolutely no pseudo-code
- Absolutely no implementation code

**What IS Allowed:**

- Reference libraries, frameworks, tools, or technologies by name
- Describe logic, architecture, flows, and mechanisms
- Explain conceptual approaches and methodologies
- Discuss design patterns and architectural decisions
- Compare different approaches or technologies
- Provide high-level system design descriptions

**Rationale:**

These constraints ensure that deep research tasks remain focused on:

1. **Analysis:** Understanding existing systems and approaches
2. **Architecture:** Designing solutions at a conceptual level
3. **Planning:** Creating detailed implementation roadmaps
4. **Education:** Explaining concepts without implementation details
5. **Documentation:** Producing comprehensive design documents

**Enforcement:**

- Any violation of these constraints invalidates the research output
- If implementation details are needed, they must be explicitly requested in a separate task
- Maintain strict separation between research/design and implementation phases

---

### Part 7: Original Task Specification

**Task Overview:**

Create a strategic, step-by-step, deeply detailed guide on how to design and build an entire AI agent—from scratch—similar in power, philosophy, and capability to:

**Inspiration Sources:**

| Project                   | Description                                                     | Repository                                                      |
| ------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------- |
| **OpenCode AI**           | Intelligent code assistance patterns and agent-driven workflows | [opencode-ai/opencode](https://github.com/opencode-ai/opencode) |
| **Crush (Charmbracelet)** | Elegant terminal UI paradigms and user experience design        | [charmbracelet/crush](https://github.com/charmbracelet/crush)   |

**Agent Identity:**

- **Name:** Proxima
- **Purpose:** Orchestrate quantum simulations across multiple backends
- **Philosophy:** Independent, extensible system with its own identity

**Target Quantum Backends:**

| Backend        | Simulator Types                                                | Repository                                                                             |
| -------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **LRET**       | Framework Integration                                          | [kunal5556/LRET](https://github.com/kunal5556/LRET/tree/feature/framework-integration) |
| **Cirq**       | Density Matrix Simulator, State Vector Simulator               | [quantumlib/Cirq](https://github.com/quantumlib/Cirq)                                  |
| **Qiskit Aer** | Density Matrix Simulator, State Vector Simulator               | [Qiskit/qiskit-aer](https://github.com/Qiskit/qiskit-aer)                              |
| **Extensible** | Additional quantum or classical backends as extensible targets | —                                                                                      |

**Core Capability:**

Proxima should allow users to seamlessly select, compare, analyze, and interpret results from these backends within a single intelligent agent workflow.

**Design Principle:**

- Take architectural, UX, and feature inspiration from OpenCode AI and Crush
- Proxima must be designed as its own independent, extensible system
- NOT a copy—original implementation with inspired patterns

**Guide Requirements:**

1. Strategic system architecture overview
2. Phased development roadmap
3. Step-by-step implementation guidance (descriptions only, no code)
4. Reference to specific libraries, frameworks, and tools
5. Detailed enough for advanced AI models to follow and implement

---

### Part 8: Reference Projects

**Primary Inspiration Sources:**

These projects serve as architectural and UX inspiration for Proxima. Study their patterns, philosophies, and implementations to inform design decisions.

**OpenCode AI**

- **Repository:** [https://github.com/opencode-ai/opencode](https://github.com/opencode-ai/opencode)
- **Value:** Intelligent code assistance patterns, agent-driven workflows, modular architecture
- **Key Patterns to Study:**
  - Agent orchestration mechanisms
  - Tool integration patterns
  - Context management strategies
  - User interaction flows

**Crush (Charmbracelet)**

- **Repository:** [https://github.com/charmbracelet/crush](https://github.com/charmbracelet/crush)
- **Value:** Elegant terminal UI paradigms, exceptional user experience design
- **Key Patterns to Study:**
  - Terminal UI/TUI design patterns (via Bubble Tea)
  - Progress visualization techniques
  - Interactive command interfaces
  - Clean output formatting

---

### Part 9: Agent Identity

**Agent Name:** Proxima

**Identity Characteristics:**

- **Independent:** Not a fork or derivative—built from scratch
- **Extensible:** Designed for future growth and plugin support
- **Intelligent:** Capable of autonomous decision-making within defined parameters
- **Transparent:** Always shows what it's doing and why
- **User-Centric:** Prioritizes user consent and clear communication

**Naming Rationale:**

- "Proxima" suggests proximity—being close to the user's needs
- Evokes Proxima Centauri—reaching toward new frontiers (quantum computing)
- Short, memorable, and distinctive

---

### Part 10: Primary Goal Definition

**GOAL:** Proxima must be capable of running quantum simulations across multiple backends.

**Supported Backends (Minimum Viable):**

**1. LRET (Framework Integration Branch)**

- **Repository:** [https://github.com/kunal5556/LRET/tree/feature/framework-integration](https://github.com/kunal5556/LRET/tree/feature/framework-integration)
- **Branch:** `feature/framework-integration`
- **Purpose:** Custom framework integration for specialized quantum workflows
- **Integration Priority:** High (primary custom backend)

**2. Cirq (Google Quantum AI)**

- **Repository:** [https://github.com/quantumlib/Cirq](https://github.com/quantumlib/Cirq)
- **Simulator Types:**
  - Density Matrix Simulator
  - State Vector Simulator
- **Integration Priority:** High (industry-standard framework)

**3. Qiskit Aer (IBM Quantum)**

- **Repository:** [https://github.com/Qiskit/qiskit-aer](https://github.com/Qiskit/qiskit-aer)
- **Simulator Types:**
  - Density Matrix Simulator
  - State Vector Simulator
- **Integration Priority:** High (industry-standard framework)

**4. Extensible Targets**

- Additional quantum backends (future)
- Classical simulation backends (future)
- Hybrid quantum-classical backends (future)
- Plugin architecture for third-party backends

**Multi-Backend Workflow:**

Proxima enables users to:

1. **Select:** Choose specific backend(s) for execution
2. **Compare:** Run identical simulations across multiple backends
3. **Analyze:** Examine differences in results, performance, and behavior
4. **Interpret:** Generate meaningful insights from raw simulation outputs

---

### Part 11: Cirq Backend Deep Dive

**Repository:** [https://github.com/quantumlib/Cirq](https://github.com/quantumlib/Cirq)

**Overview:**

Cirq is a Python framework for creating, editing, and invoking Noisy Intermediate Scale Quantum (NISQ) circuits developed by Google Quantum AI.

**Simulator Types for Proxima Integration:**

**1. Density Matrix Simulator**

- **Purpose:** Simulates quantum circuits using density matrix representation
- **Use Cases:**
  - Mixed state simulations
  - Noise modeling and analysis
  - Open quantum system dynamics
  - Decoherence studies
- **Advantages:**
  - Accurate representation of noisy quantum systems
  - Can model partial trace operations
  - Suitable for realistic hardware simulation
- **Limitations:**
  - Memory scales as $O(4^n)$ for $n$ qubits
  - Slower than state vector for pure states

**2. State Vector Simulator**

- **Purpose:** Simulates quantum circuits using pure state vector representation
- **Use Cases:**
  - Ideal (noiseless) circuit simulation
  - Algorithm development and testing
  - Educational purposes
  - Benchmarking
- **Advantages:**
  - Memory scales as $O(2^n)$ for $n$ qubits
  - Faster execution for pure state evolution
  - Simpler output interpretation
- **Limitations:**
  - Cannot directly model mixed states or noise
  - Limited to pure quantum states

**Cirq Integration Considerations:**

- Native Python library—direct import capability
- Rich gate library and circuit manipulation tools
- Built-in visualization capabilities
- Supports custom gate definitions
- Integration with Google's quantum hardware

---

### Part 12: Qiskit Aer Backend Deep Dive

**Repository:** [https://github.com/Qiskit/qiskit-aer](https://github.com/Qiskit/qiskit-aer)

**Overview:**

Qiskit Aer is a high-performance quantum circuit simulator framework developed by IBM Quantum, providing various simulation methods for quantum circuits.

**Simulator Types for Proxima Integration:**

**1. Density Matrix Simulator**

- **Purpose:** Full density matrix simulation for mixed quantum states
- **Use Cases:**
  - Noise simulation with realistic error models
  - Quantum channel analysis
  - Quantum error correction studies
  - Open system dynamics
- **Advantages:**
  - Comprehensive noise model support
  - Integration with IBM Quantum hardware noise models
  - Supports all Qiskit noise primitives
- **Limitations:**
  - Memory intensive: $O(4^n)$ scaling
  - Computationally expensive for large circuits

**2. State Vector Simulator**

- **Purpose:** Ideal state vector simulation for pure quantum states
- **Use Cases:**
  - Algorithm prototyping
  - Exact amplitude computation
  - Verification of quantum algorithms
  - Large-scale noiseless simulation
- **Advantages:**
  - Highly optimized C++ backend
  - GPU acceleration support
  - Memory efficient: $O(2^n)$ scaling
  - Fast execution
- **Limitations:**
  - No native noise support (pure states only)
  - Requires additional configuration for noise injection

**Qiskit Aer Integration Considerations:**

- High-performance C++ core with Python bindings
- GPU acceleration via CUDA (optional)
- Extensive noise modeling library
- Direct integration with IBM Quantum Experience
- Supports OpenQASM circuit format
- Rich transpilation and optimization pipelines

---

### Part 13: Extensible Backend Architecture

**Purpose:** Enable Proxima to support additional quantum or classical backends beyond the core three (LRET, Cirq, Qiskit Aer).

**Extensibility Goals:**

1. **Future Quantum Backends:**

   - PennyLane (Xanadu)
   - Amazon Braket SDK
   - Azure Quantum SDK
   - Rigetti Forest/pyQuil
   - IonQ native SDK
   - Strangeworks
   - ProjectQ

2. **Classical Simulation Backends:**

   - QuTiP (Quantum Toolbox in Python)
   - TensorNetwork (Google)
   - ITensor
   - Custom matrix operation libraries

3. **Hybrid Quantum-Classical Backends:**
   - Variational algorithm executors
   - Quantum machine learning frameworks
   - Quantum-inspired classical algorithms

**Plugin Architecture Design:**

**Backend Interface Contract:**

Every backend adapter must implement:

- **Initialize:** Set up backend connection and configuration
- **Validate Circuit:** Check circuit compatibility with backend
- **Execute:** Run the quantum circuit/simulation
- **Retrieve Results:** Get and format execution results
- **Capabilities Query:** Report supported features and limitations
- **Resource Estimation:** Estimate memory, time, and compute requirements

**Registration Mechanism:**

- Plugin discovery via entry points or configuration files
- Runtime backend loading without core code changes
- Version compatibility checking
- Graceful degradation for unavailable backends

**Adapter Pattern:**

Each backend adapter translates between:

- Proxima's internal circuit representation
- Backend-specific circuit format
- Standardized result format for comparison

**Benefits of Extensibility:**

1. **Future-Proofing:** New backends can be added without core rewrites
2. **Community Contributions:** Third-party developers can create adapters
3. **Specialization:** Users can add domain-specific backends
4. **Hardware Evolution:** Easily integrate new quantum hardware as it becomes available
5. **Comparison Studies:** Run identical experiments across many backends

---

### Part 14: Unified Agent Workflow

**Core Capability Statement:**

Proxima should allow users to seamlessly select, compare, analyze, and interpret results from these backends within a single intelligent agent workflow.

**Workflow Components:**

**1. Seamless Selection**

- Users can choose one or multiple backends for execution
- Backend selection can be explicit (user-specified) or automatic (agent-recommended)
- No context switching required between different backend interfaces
- Unified command syntax regardless of target backend

**2. Comparison Capabilities**

- Execute identical quantum circuits across multiple backends simultaneously
- Side-by-side result visualization
- Automated difference detection and highlighting
- Performance metrics comparison (execution time, resource usage)

**3. Analysis Features**

- Statistical analysis of result distributions
- Fidelity calculations between backend outputs
- Error analysis and noise characterization
- Trend identification across multiple runs

**4. Interpretation Layer**

- Transform raw quantum data into actionable insights
- Natural language explanations of results
- Visualization generation (histograms, Bloch spheres, density matrices)
- Recommendation engine for next steps

**Single Workflow Benefits:**

- Eliminates need to learn multiple CLI/API interfaces
- Consistent experience regardless of backend
- Reduced cognitive load for users
- Streamlined research and development cycles

---

### Part 15: Inspiration Philosophy

**INSPIRATION (NOT COPYING)**

**Guiding Principle:**

You may take architectural, UX, and feature inspiration from OpenCode AI and Crush, but Proxima must be designed as its own independent, extensible system.

**What to Learn From OpenCode AI:**

- Agent orchestration patterns
- Context management strategies
- Tool integration approaches
- Conversation flow design
- Error handling paradigms

**What to Learn From Crush (Charmbracelet):**

- Terminal UI excellence
- Progress visualization
- Interactive prompts
- Clean output formatting
- User feedback mechanisms

**What Makes Proxima Unique:**

1. **Domain Focus:** Quantum simulation orchestration (not general code assistance)
2. **Multi-Backend Architecture:** First-class support for backend comparison
3. **Scientific Workflow:** Designed for research and experimentation
4. **Resource Awareness:** Deep integration with hardware monitoring
5. **Consent-First Design:** Explicit permission model for all significant actions

**Independence Requirements:**

- Original codebase—no forking or direct code copying
- Distinct branding and identity
- Unique architectural decisions driven by quantum workflow needs
- Custom feature set tailored to simulation use cases
- Separate development roadmap and versioning

---

### Part 16: Mandatory Features Overview

**MANDATORY FEATURES (NON-NEGOTIABLE)**

The following features are absolute requirements for Proxima. No compromise is acceptable on these capabilities.

**Feature Categories:**

| #   | Feature                                 | Priority | Status   |
| --- | --------------------------------------- | -------- | -------- |
| 1   | Execution Timer & Transparency          | Critical | Required |
| 2   | Backend Selection & Intelligence        | Critical | Required |
| 3   | Fail-Safe & Resource Awareness          | Critical | Required |
| 4   | Execution Control                       | Critical | Required |
| 5   | Result Interpretation & Insights        | Critical | Required |
| 6   | Multi-Backend Comparison                | Critical | Required |
| 7   | Planning, Analysis & Execution Pipeline | Critical | Required |
| 8   | API Key & Local LLM Integration         | Critical | Required |
| 9   | proxima_agent.md Compatibility          | Critical | Required |
| 10  | Additional Inspired Features            | High     | Required |
| 11  | Future UI Planning                      | Medium   | Planned  |

**Non-Negotiable Meaning:**

- These features define Proxima's core value proposition
- Missing any feature renders the system incomplete
- Each feature must be fully functional before release
- Quality standards cannot be lowered for any feature

---

### Part 17: Execution Timer & Transparency

**Feature Requirement:**

The agent must clearly display execution information at all times during operation.

**Required Display Elements:**

**1. Current Running Task/Process**

- Display what code, process, or task is currently executing
- Show the exact operation being performed
- Include relevant context (file names, function names, backend being used)
- Update in real-time as execution progresses

**2. Elapsed Time Tracking**

- Show how long the current task has been running
- Display in human-readable format (e.g., "2m 34s" or "00:02:34")
- Update continuously during execution
- Provide time estimates when possible (e.g., "~3 minutes remaining")

**3. Stage-Level Execution Tracking**

- Break down complex operations into visible stages
- Show progress through each stage
- Indicate current stage vs. total stages (e.g., "Stage 3/7: Executing Circuit")
- Provide stage-specific timing information

**Implementation Requirements:**

**Visual Indicators:**

- Spinner or progress animation for active tasks
- Progress bars where percentage completion is determinable
- Stage completion checkmarks
- Time counters with sub-second precision when appropriate

**Transparency Levels:**

| Level    | Information Displayed                                                |
| -------- | -------------------------------------------------------------------- |
| Minimal  | Current task name, elapsed time                                      |
| Standard | Task name, elapsed time, current stage, progress indicator           |
| Verbose  | All above + detailed operation logs, resource usage, backend details |

**User Configuration:**

- Allow users to set preferred transparency level
- Provide command to toggle between levels during execution
- Remember user preference across sessions

**Example Display States:**

**During Circuit Execution:**

```
[Running] Executing quantum circuit on Cirq (State Vector)
├── Stage: Circuit Compilation [✓] (1.2s)
├── Stage: Simulation Setup [✓] (0.3s)
├── Stage: Execution [▶] (12.4s elapsed)
│   └── Progress: 67% (1024/1536 shots)
└── Elapsed: 13.9s | Est. remaining: ~7s
```

**During Multi-Backend Comparison:**

```
[Comparing] Running circuit across 3 backends
├── Cirq (State Vector) [✓] Completed (4.2s)
├── Qiskit Aer (State Vector) [✓] Completed (3.8s)
├── LRET [▶] Running (2.1s elapsed)
└── Total elapsed: 6.3s
```

**Rationale:**

- Users must never wonder "what is happening?"
- Provides confidence that the system is working
- Enables informed decisions about aborting long operations
- Supports debugging and performance analysis
- Builds trust through transparency

---

### Part 18: Backend Selection & Intelligence

**Feature Requirement #2:**

The user must have full freedom to explicitly choose a backend, with intelligent automatic selection as a fallback.

**User Freedom (Explicit Selection):**

- Users can specify exact backend for any operation
- Selection options include:
  - Backend type (LRET, Cirq, Qiskit Aer, custom)
  - Simulator variant (State Vector, Density Matrix)
  - Specific configuration parameters
- User choice always takes precedence over automatic selection
- No hidden overrides or "we know better" behavior

**Automatic Selection (When User Does Not Choose):**

If the user does NOT explicitly choose a backend, Proxima must:

**1. Analyze the Query or Workload**

- Parse the quantum circuit or simulation request
- Identify circuit characteristics:
  - Number of qubits
  - Gate types and depth
  - Measurement requirements
  - Noise model requirements
  - Precision requirements
- Evaluate workload constraints:
  - Available memory
  - Time constraints
  - Accuracy requirements

**2. Automatically Select the Most Suitable Backend**

- Apply selection algorithm based on analysis:
  - Small circuits (< 10 qubits) → Prefer fastest simulator
  - Noise simulation required → Prefer Density Matrix simulators
  - Large pure-state circuits → Prefer optimized State Vector simulators
  - Custom framework needs → Prefer LRET
  - GPU acceleration needed → Prefer backends with GPU support

**3. Explain Why That Backend Was Chosen**

- Provide clear, natural language explanation
- Example: "Selected Qiskit Aer State Vector Simulator because: (1) Your circuit has 8 qubits with pure-state evolution, (2) No noise model was specified, (3) This backend offers the fastest execution for your circuit depth."

**Selection Transparency Requirements:**

| Scenario                           | Required Behavior                               |
| ---------------------------------- | ----------------------------------------------- |
| User specifies backend             | Use exactly as specified, confirm selection     |
| User doesn't specify               | Auto-select, explain reasoning before execution |
| Auto-selected backend unavailable  | Inform user, suggest alternatives               |
| Multiple backends equally suitable | Present options, ask user to choose             |

---

### Part 19: Fail-Safe, Resource Awareness & Explicit Consent

**Feature Requirement #3:**

Proxima must detect, handle, and communicate resource constraints with explicit user consent for risky operations.

**Resource Detection Requirements:**

Proxima must detect and handle:

**1. Low Memory Conditions**

- Monitor available RAM continuously during operation
- Threshold warnings at 80%, 90%, 95% memory usage
- Pre-execution estimation of memory requirements
- Comparison of required vs. available memory

**2. Memory Exhaustion**

- Detect approaching out-of-memory conditions
- Graceful degradation before crash
- Safe state preservation when possible
- Clear error messaging if exhaustion occurs

**3. RAM Limits**

- Detect system RAM capacity
- Calculate per-backend memory requirements
- Scale recommendations based on available resources
- Suggest circuit simplification if needed

**4. Hardware Incompatibility**

- GPU availability detection for CUDA-accelerated backends
- CPU instruction set compatibility
- Operating system compatibility
- Python version and dependency compatibility

**Silent Action Prohibition:**

- **No action may occur silently**
- All significant operations require user acknowledgment
- All warnings must be displayed prominently
- All risks must be explained before proceeding

**Hardware Incompatibility Protocol:**

If user hardware is incompatible, Proxima must:

**Step 1: Clearly Warn the User**

```
⚠️ WARNING: Hardware Incompatibility Detected
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your system may not be able to execute this operation safely.
```

**Step 2: Explain Risks and Limitations**

```
DETECTED ISSUES:
• Insufficient RAM: 4GB available, ~6GB required
• No CUDA GPU: GPU acceleration unavailable

POTENTIAL RISKS:
• System may become unresponsive
• Execution may fail mid-computation
• Results may be incomplete or corrupted
```

**Step 3: Offer Explicit "Force Execute Anyway" Option**

```
OPTIONS:
[1] Cancel operation (recommended)
[2] Reduce circuit size to fit resources
[3] Force execute anyway (NOT RECOMMENDED)
    └── Requires explicit confirmation: Type "FORCE EXECUTE"
```

**Step 4: Proceed Only After Direct, Explicit User Consent**

- Passive defaults (hitting Enter) must NOT proceed with risky operations
- Require active confirmation (typing specific phrase)
- Log consent for accountability

---

### Part 20: Execution Control

**Feature Requirement #4:**

Proxima must provide comprehensive execution control with visible state transitions.

**Required Control Operations:**

**1. Start**

- Initiate execution of planned task
- Pre-execution validation checks
- Resource allocation and setup
- Clear confirmation of execution beginning

**2. Abort**

- Immediately stop current execution
- Clean up partial resources
- Report execution state at abort time
- No data corruption from abort

**3. Rollback (If Feasible, But Preferred)**

- Restore system to pre-execution state
- Undo partial changes where possible
- Clear indication of what was rolled back
- Explanation if rollback is not possible

**4. Pause (If Feasible)**

- Suspend execution without losing progress
- Preserve current state in memory
- Report pause status and resources held
- Allow inspection during pause

**5. Resume (If Feasible)**

- Continue from paused state
- Verify state integrity before resuming
- Report resumed position
- Handle timeout scenarios

**State Transition Visibility:**

All state transitions must be visible and traceable:

```
EXECUTION STATE MACHINE:

    ┌──────────┐
    │   IDLE   │ ◄────────────────┐
    └────┬─────┘                  │
         │ start                  │
         ▼                        │
    ┌──────────┐     pause   ┌────┴─────┐
    │ RUNNING  │ ───────────►│  PAUSED  │
    └────┬─────┘             └────┬─────┘
         │                        │ resume
         │ ◄──────────────────────┘
         │
    ┌────┴────┬─────────────┐
    │         │             │
    ▼         ▼             ▼
┌───────┐ ┌───────┐ ┌───────────┐
│COMPLETE│ │ABORTED│ │  FAILED   │
└───────┘ └───────┘ └───────────┘
         │
         ▼
    ┌──────────┐
    │ ROLLBACK │
    └──────────┘
```

**State Logging:**

- Every transition logged with timestamp
- Reason for transition recorded
- State history accessible to user
- Export capability for debugging

---

### Part 21: Result Interpretation & Insights

**Feature Requirement #5:**

Proxima must generate clear, meaningful insights from simulation outputs—not raw data dumps.

**Supported Output Formats:**

- `.xlsx` files (Excel spreadsheets)
- `.csv` files (Comma-separated values)
- JSON result objects
- Native backend result formats

**Insight Quality Requirements:**

**1. Human-Readable**

- Clear language, minimal jargon
- Proper formatting and structure
- Visual elements where helpful (tables, summaries)
- Appropriate level of detail based on user expertise

**2. Analytical**

- Statistical summaries (mean, variance, distribution)
- Pattern identification
- Anomaly detection
- Trend analysis across multiple runs

**3. Decision-Oriented**

- Actionable conclusions
- Recommendations for next steps
- Confidence levels for findings
- Trade-off analysis when relevant

**4. Not Raw Data Dumps**

- Never output unprocessed arrays or matrices
- Always contextualize numerical results
- Explain what numbers mean
- Highlight significant findings

**Insight Generation Pipeline:**

```
Raw Results → Parsing → Statistical Analysis → Pattern Detection →
Insight Formulation → Natural Language Generation → User Presentation
```

**Example Insight Output:**

Instead of:

```
[0.498, 0.502, 0.001, 0.001, ...]
```

Provide:

```
SIMULATION INSIGHTS
═══════════════════

📊 Measurement Distribution Analysis
   • Dominant states: |00⟩ (49.8%), |01⟩ (50.2%)
   • Near-zero probability states: |10⟩, |11⟩ (< 0.1% each)

🎯 Key Finding
   Your circuit produces an approximately equal superposition
   of the first two basis states, consistent with a Hadamard
   gate applied to the first qubit.

📈 Statistical Summary
   • Shots: 1024
   • Entropy: 0.999 bits (max: 1.0)
   • Fidelity to ideal: 99.7%

💡 Recommendation
   Results show high fidelity. Consider increasing shot count
   for more precise probability estimates if needed.
```

---

### Part 22: Multi-Backend Comparison

**Feature Requirement #6:**

Enable running and comparing identical simulations across multiple backends within a single execution.

**Comparison Capabilities:**

**1. Multiple Backends**

- Select 2 or more backends for comparison
- Include any combination of supported backends
- Support mixed simulator types

**2. Identical Parameters**

- Ensure exact same circuit is executed
- Use identical shot counts
- Apply equivalent noise models where applicable
- Control for random seed when possible

**3. Within Same Execution Run**

- Single command triggers all backend executions
- Coordinated parallel or sequential execution
- Unified progress tracking
- Combined result collection

**4. Structured Comparative Analysis**

Provide comparison across:

| Metric                  | Cirq SV         | Qiskit Aer SV | LRET  |
| ----------------------- | --------------- | ------------- | ----- | ----- |
| Execution Time          | 4.2s            | 3.8s          | 5.1s  |
| Memory Usage            | 128MB           | 142MB         | 98MB  |
|                         | 00⟩ Probability | 0.498         | 0.501 | 0.499 |
|                         | 01⟩ Probability | 0.502         | 0.499 | 0.501 |
| Fidelity (vs reference) | 99.9%           | 99.8%         | 99.7% |

**Comparison Report Sections:**

1. **Performance Comparison:** Time, memory, throughput
2. **Result Accuracy:** Probability distributions, fidelity metrics
3. **Discrepancy Analysis:** Where backends differ and why
4. **Recommendation:** Which backend suits the use case best

---

### Part 23: Planning, Analysis & Execution Pipeline

**Feature Requirement #7:**

Proxima must follow a structured pipeline with transparent stages.

**Pipeline Stages:**

**Stage 1: Plan the Task**

- Parse user request
- Identify required operations
- Determine resource requirements
- Create execution plan
- Present plan to user for approval

**Stage 2: Analyze Requirements**

- Validate circuit syntax and semantics
- Check backend compatibility
- Assess resource availability
- Identify potential issues
- Generate risk assessment

**Stage 3: Execute in Structured Stages**

- Execute plan step-by-step
- Report progress at each stage
- Handle errors gracefully
- Maintain execution log

**Transparency Requirements:**

Each stage must be:

- **Visible:** User can see current stage
- **Explainable:** User can understand why this stage is needed
- **Interruptible:** User can pause or abort between stages
- **Logged:** Complete record maintained

**Example Pipeline Display:**

```
EXECUTION PIPELINE
══════════════════

[✓] Stage 1: Planning
    └── Created execution plan for 3-qubit GHZ circuit

[✓] Stage 2: Analysis
    └── Validated circuit, estimated 45MB memory requirement

[▶] Stage 3: Execution
    ├── Substage 3.1: Backend initialization [✓]
    ├── Substage 3.2: Circuit compilation [✓]
    ├── Substage 3.3: Simulation running [▶] 67%
    └── Substage 3.4: Result collection [pending]

[ ] Stage 4: Interpretation
    └── Waiting for execution completion
```

---

### Part 24: API Key & Local LLM Integration

**Feature Requirement #8:**

Support both API-based and local LLM integration with explicit consent protocols.

**API Key Integration:**

- Support for major LLM providers (OpenAI, Anthropic, etc.)
- Secure API key storage and management
- Key validation before use
- Usage tracking and reporting

**LLM-Assisted Backend Modification:**

- Proxima can modify backend behavior based on user requirements
- Modifications guided by LLM analysis of user intent
- **Requires explicit user permission before any modification**

**Local LLM Integration:**

If the user has a locally installed LLM, Proxima must:

**1. Offer as Execution Option**

```
LLM OPTIONS DETECTED
════════════════════
[1] Use local LLM: Ollama (llama3:8b)
[2] Use remote API: OpenAI GPT-4
[3] Proceed without LLM assistance
```

**2. Allow LLM to Assist In:**

- **Modifying backend logic (conceptually):** Suggest parameter changes, optimization strategies
- **Explaining results:** Generate natural language interpretations
- **Analyzing outputs:** Identify patterns, anomalies, insights

**3. Clearly Distinguish Between:**

| Indicator | Meaning                                    |
| --------- | ------------------------------------------ |
| 🏠 LOCAL  | Using local LLM (data stays on device)     |
| 🌐 REMOTE | Using API-based LLM (data sent externally) |

**4. Explicit Consent Requirements:**

- Always require user consent before invoking any LLM
- Separate consent for local vs. remote
- Explain what data will be processed
- Allow per-session or persistent consent

**Consent Dialog Example:**

```
LLM ASSISTANCE REQUEST
══════════════════════

Proxima would like to use an LLM to analyze your simulation results.

📤 Data to be processed:
   • Measurement probabilities (1024 values)
   • Circuit description (15 gates)

🌐 Using: OpenAI GPT-4 (remote API)
   └── Data will be sent to OpenAI servers

[Y] Yes, proceed with LLM analysis
[N] No, skip LLM analysis
[L] Switch to local LLM instead
```

---

### Part 25: proxima_agent.md Compatibility

**Feature Requirement #9:**

Design architecture to support future consumption of instruction files.

**Future Capability:**

Proxima must be designed so it can later:

- **Consume:** Read and parse `proxima_agent.md` files
- **Interpret:** Understand structured instructions within the file
- **Execute:** Carry out instructions as autonomous workflows

**Architecture Requirements:**

**1. Instruction Parser Interface**

- Define abstract interface for instruction parsing
- Allow pluggable parsers for different formats
- Support Markdown-based instruction format

**2. Workflow Engine**

- Task queue management
- Conditional execution support
- Loop and iteration handling
- Error recovery mechanisms

**3. Instruction Schema (Future Definition)**

Anticipated structure:

```markdown
# proxima_agent.md

## Task: Run Bell State Experiment

### Parameters

- qubits: 2
- shots: 4096
- backends: [cirq_sv, qiskit_sv]

### Steps

1. Create Bell state circuit
2. Execute on all specified backends
3. Compare results
4. Generate report

### On Error

- Retry up to 3 times
- If persistent, notify user
```

**Current Implementation:**

- Design all components with instruction-driven execution in mind
- Use command pattern for all operations
- Maintain operation history for replay capability
- Build modular, composable workflow primitives

---

### Part 26: Additional Inspired Features

**Feature Requirement #10:**

Include valuable features inspired by OpenCode AI and Crush.

**Feature 1: Conversation Memory & Context**

_Inspired by: OpenCode AI_

- Maintain context across multiple interactions
- Remember previous circuits, results, preferences
- Enable follow-up queries ("Run that again with more shots")
- **User Value:** Reduces repetition, enables iterative exploration
- **System Robustness:** Improves accuracy of intent understanding

**Feature 2: Interactive REPL Mode**

_Inspired by: Crush (Charmbracelet)_

- Real-time quantum circuit building
- Immediate feedback on circuit validity
- Quick execution of small experiments
- **User Value:** Rapid prototyping and learning
- **System Robustness:** Catches errors early

**Feature 3: Rich Terminal Formatting**

_Inspired by: Crush (Charmbracelet)_

- Beautiful, readable output using terminal styling
- Tables, progress bars, spinners, colors
- Adaptive to terminal capabilities
- **User Value:** Pleasant, professional experience
- **System Robustness:** Clear communication reduces confusion

**Feature 4: Command History & Replay**

_Inspired by: OpenCode AI_

- Save all executed commands
- Replay previous operations
- Share command sequences
- **User Value:** Reproducibility, collaboration
- **System Robustness:** Debugging, audit trail

**Feature 5: Configuration Profiles**

_Inspired by: Both projects_

- Save named configurations (backend preferences, defaults)
- Quick switching between profiles
- Import/export configurations
- **User Value:** Workflow efficiency
- **System Robustness:** Consistent environments

**Feature 6: Plugin Ecosystem**

_Inspired by: OpenCode AI architecture_

- Extensible plugin interface
- Community-contributed backends and tools
- Safe sandbox execution for plugins
- **User Value:** Customization, extended capabilities
- **System Robustness:** Modular, maintainable codebase

---

### Part 27: Future UI Considerations

**Feature Requirement #11:**

Acknowledge UI importance while deferring detailed implementation.

**Ultimate UI Goals:**

Proxima should ultimately have:

- **Smooth:** Responsive, no lag or jank
- **Clean:** Minimal, uncluttered interface
- **Modern:** Contemporary design patterns
- **Good-looking:** Aesthetically pleasing

**Current Priority:**

- UI design is **NOT the main focus for now**
- Treat UI considerations as **future work only**
- Focus on CLI/TUI excellence first

**Future UI Roadmap:**

| Phase     | UI Focus                      |
| --------- | ----------------------------- |
| Phase 1-4 | CLI only, no UI work          |
| Phase 5   | TUI enhancements (Bubble Tea) |
| Phase 6+  | Web UI exploration (optional) |

**UI Technology Considerations (Future):**

- **TUI:** Bubble Tea (Go), Rich/Textual (Python)
- **Web:** React, Vue, or Svelte frontend
- **Desktop:** Electron, Tauri, or native

---

### Part 28: Response Structure Requirements

**Document Organization (STRICT & ENFORCED):**

This guide MUST be long, exhaustive, and structured exactly as follows:

**Required Sections:**

1. Strategic System Sketch
2. Phased Roadmap
3. Phase-by-Phase Implementation Guide
4. Phase Summaries & Usage Guidance

**Quality Standards:**

- Comprehensive coverage of all topics
- Clear, logical organization
- Actionable guidance throughout
- No gaps in information

---

### Part 29: Strategic System Sketch Details

**Required Content:**

**1. Overall Architecture**

- Layer diagram (User Interface → Orchestration → Intelligence → Resources → Backends → Data)
- Component relationships
- Dependency structure

**2. Core Components**

- Each major module defined
- Responsibilities clearly stated
- Interfaces between components

**3. Data, Control, and Decision Flow**

- How data moves through the system
- Control flow for operations
- Decision points and logic

**4. How All Features Interconnect**

- Feature dependency map
- Shared services identification
- Integration points

---

### Part 30: Phased Roadmap Structure

**Requirements:**

**1. Divide Journey into Clear, Logical Phases**

- Each phase has distinct goals
- Phases build upon each other
- Clear entry and exit criteria

**2. Each Phase Moves Proxima Closer to Final System**

- Progressive capability addition
- Testable milestones
- Demonstrable progress

**Recommended Phase Structure:**

| Phase | Focus           | Outcome                      |
| ----- | --------------- | ---------------------------- |
| 1     | Foundation      | Core infrastructure          |
| 2     | Basic Execution | Single backend working       |
| 3     | Intelligence    | Auto-selection, insights     |
| 4     | Multi-Backend   | Comparison capabilities      |
| 5     | LLM Integration | AI-assisted features         |
| 6     | Polish          | UI, documentation, stability |

---

### Part 31: Phase Implementation Guide Requirements

**For Each Phase, Provide:**

**1. Step-by-Step Technical Instructions**

- Ordered sequence of implementation steps
- Dependencies between steps identified
- Clear completion criteria

**2. Descriptions Only (No Code)**

- Describe what to implement
- Explain the logic and approach
- Reference patterns and techniques

**3. Use Names of Libraries, Frameworks, and Tools**

- Specific technology recommendations
- Version considerations where relevant
- Alternative options noted

**4. Detailed Enough for Advanced AI to Follow**

- GPT-5.1 Codex Max or Claude Opus 4.5 should be able to implement directly
- No ambiguity in instructions
- Complete information provided

**5. No Missing Steps, No Vague Guidance**

- Every necessary step included
- Specific rather than general
- Actionable instructions

---

### Part 32: Phase Summaries & Usage Requirements

**For Each Phase, Provide:**

**1. Summarize Features Implemented**

- List of capabilities added
- What's now possible that wasn't before
- Technical achievements

**2. Describe New System Capabilities**

- User-facing functionality
- Internal improvements
- Performance characteristics

**3. Provide Clear Instructions and Operational Guidance**

- How to use new features
- Example commands/workflows
- Common scenarios addressed

**Summary Format Example:**

```
PHASE X SUMMARY
═══════════════

✅ Features Implemented:
   • Feature A: [description]
   • Feature B: [description]

🚀 New Capabilities:
   • Users can now [capability]
   • System now supports [capability]

📖 Usage Guide:
   To use Feature A:
   1. [Step 1]
   2. [Step 2]

   Example: [concrete example]
```

---

_Document generated for the Proxima AI Agent project. This is a living document—update as the project evolves._
