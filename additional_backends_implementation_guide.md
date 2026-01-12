# Proxima: Additional Backends Implementation Guide

> **Document Type:** Strategic Implementation Guide  
> **Version:** 1.0  
> **Date:** January 12, 2026  
> **Purpose:** Detailed guide for integrating QuEST, cuQuantum, and qsim backends into Proxima  
> **Target Audience:** AI coding agents (GPT-5.1 Codex Max, Opus 4.5) and developers

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1: QuEST Integration](#phase-1-quest-integration)
4. [Phase 2: cuQuantum Integration](#phase-2-cuquantum-integration)
5. [Phase 3: qsim Integration](#phase-3-qsim-integration)
6. [Phase 4: Unified Backend Selection Enhancement](#phase-4-unified-backend-selection-enhancement)
7. [Phase 5: Testing & Validation](#phase-5-testing--validation)
8. [Phase 6: Documentation & Deployment](#phase-6-documentation--deployment)

---

## Overview

### New Backends Summary

| Backend    | Type               | Primary Use Case            | Hardware Support     | Integration Method       |
|------------|--------------------|-----------------------------|----------------------|--------------------------|
| **QuEST**  | C++ Simulator      | High-performance DM + SV    | CPU, GPU (CUDA/HIP)  | Python bindings          |
| **cuQuantum** | NVIDIA GPU Library | GPU-accelerated SV          | NVIDIA GPUs only     | Qiskit Aer backend option |
| **qsim**   | Google Simulator   | High-performance SV (CPU)   | CPU (AVX/FMA)        | qsimcirq + Cirq adapter  |

### Architecture Integration Points

```
Existing Proxima Architecture:
├── backends/
│   ├── base.py              (No changes needed)
│   ├── registry.py          (Minor updates for new backends)
│   ├── lret.py              (Existing - no changes)
│   ├── cirq_adapter.py      (Extend for qsim support)
│   ├── qiskit_adapter.py    (Extend for cuQuantum support)
│   └── NEW FILES:
│       ├── quest_adapter.py        ← NEW
│       ├── cuquantum_adapter.py    ← NEW (extends qiskit_adapter)
│       └── qsim_adapter.py         ← NEW (leverages cirq_adapter)
```

---

## Prerequisites

### System Requirements

**For QuEST:**
- C++ compiler with C++11 support (gcc 5.0+ or MSVC 2015+)
- CMake 3.16 or higher
- Eigen3 library for linear algebra
- Optional: CUDA Toolkit 11.2+ for GPU support
- Optional: OpenMP for parallel execution
- Optional: MPI for distributed computing

**For cuQuantum:**
- NVIDIA GPU with compute capability 7.0+ (Volta or newer)
- CUDA Toolkit 12.0 or 13.0
- cuQuantum SDK (cuStateVec library)
- Qiskit Aer GPU package (qiskit-aer-gpu or qiskit-aer-gpu-cu11)
- NVIDIA driver supporting CUDA version

**For qsim:**
- C++17 compiler
- AVX2 or AVX512 CPU support (Intel/AMD)
- OpenMP for parallelization
- qsimcirq Python package
- Cirq (already installed as Proxima dependency)

### Python Dependencies to Add

Update Proxima's `pyproject.toml` or `requirements.txt`:

```
New dependencies:
- pyquest-cffi>=0.9.0 (QuEST Python bindings)
- qiskit-aer-gpu>=0.13.0 (for cuQuantum support)
- qsimcirq>=0.22.0 (qsim Python interface)
- custatevec-cu12>=1.6.0 (cuQuantum Python bindings - optional)
```

---

## Phase 1: QuEST Integration

**Duration:** 2-3 weeks  
**Goal:** Create fully functional QuEST adapter supporting both Density Matrix and State Vector modes

---

### Step 1.1: Understand QuEST Architecture

**Research Tasks:**

1. Study QuEST documentation from GitHub repository (https://github.com/QuEST-Kit/QuEST)
2. Identify core QuEST functions: `createQuESTEnv`, `createQureg`, `createDensityQureg`, `destroyQureg`, `destroyQuESTEnv`
3. Understand QuEST gate operations: single-qubit gates, controlled gates, multi-qubit gates
4. Review QuEST measurement functions: `measure`, `measureWithStats`, `calcProbOfOutcome`
5. Examine QuEST's Python bindings (pyQuEST or pyquest-cffi)

**Key Concepts to Map:**

- QuEST's `Qureg` object → Proxima's circuit representation
- QuEST's environment setup → Proxima's backend initialization
- QuEST gate names → Cirq/Qiskit gate equivalents
- QuEST measurement outcomes → Proxima's ExecutionResult format

---

### Step 1.2: Install and Verify QuEST

**Installation Process:**

1. **Option A: Install pre-built Python bindings**
   - Use pip to install pyquest-cffi package
   - Verify installation by importing pyquest module
   - Test basic functionality with simple circuit

2. **Option B: Build from source (for GPU support)**
   - Clone QuEST repository from GitHub
   - Configure CMakeLists.txt for GPU support (enable CUDA, cuQuantum)
   - Build QuEST shared library using CMake
   - Install Python bindings using setup.py or build custom wrapper
   - Verify GPU functionality with CUDA sample programs

**Verification Checklist:**

- Import pyquest successfully without errors
- Create QuEST environment and qureg objects
- Apply basic gates (Hadamard, CNOT, phase gates)
- Perform measurements and retrieve results
- Verify memory cleanup (no leaks)

---

### Step 1.3: Create QuEST Adapter Class

**File Location:** `src/proxima/backends/quest_adapter.py`

**Class Structure:**

```
QuestAdapter (extends BackendAdapter base class)
├── __init__()
│   └── Initialize QuEST environment, detect GPU availability
├── get_name() → "quest"
├── get_version() → Query QuEST library version
├── get_capabilities()
│   └── Return: supports_density_matrix=True, supports_statevector=True,
│                supports_gpu=True/False, max_qubits=30 (or based on RAM)
├── validate_circuit(circuit)
│   └── Check circuit compatibility with QuEST gate set
├── estimate_resources(circuit)
│   └── Calculate memory: 2^n qubits * precision * rank (for DM: 2^(2n))
├── execute(circuit, options)
│   └── Main execution logic
└── _cleanup()
    └── Destroy qureg and QuEST environment
```

**Implementation Details:**

1. **Initialization Logic:**
   - Create QuEST environment using `createQuESTEnv`
   - Detect available hardware (CPU cores, GPU devices)
   - Store environment handle for reuse
   - Configure precision mode (single, double, quad precision)

2. **Circuit Translation:**
   - Convert Proxima's circuit representation to QuEST gate sequence
   - Map Cirq/Qiskit gates to QuEST equivalents:
     - `cirq.H` → `hadamard(qureg, qubit_idx)`
     - `cirq.CNOT` → `controlledNot(qureg, control_idx, target_idx)`
     - `cirq.Rz(θ)` → `rotateZ(qureg, qubit_idx, theta)`
     - Custom gates → Decompose into QuEST-supported gates
   - Handle parameterized gates with angle conversions

3. **Qureg Creation Logic:**
   - For State Vector mode:
     - Use `createQureg(num_qubits, env)`
     - Initialize to |0⟩ state or custom initial state
   - For Density Matrix mode:
     - Use `createDensityQureg(num_qubits, env)`
     - Initialize to |0⟩⟨0| or mixed state

4. **Gate Application:**
   - Iterate through circuit gate sequence
   - Apply each gate to qureg using QuEST functions
   - Handle gate parameters (angles, targets, controls)
   - Support multi-qubit gates with proper indexing

5. **Measurement Implementation:**
   - For shot-based measurements:
     - Use `measure(qureg, qubit_idx)` in loop for each shot
     - Collapse state after measurement (statevector mode)
     - Aggregate measurement outcomes into counts dictionary
   - For statevector retrieval:
     - Use `getStateVector()` to extract amplitudes
     - Convert to numpy array format
   - For density matrix retrieval:
     - Use `getDensityMatrix()` to extract matrix elements
     - Convert to numpy array (2^n × 2^n matrix)

6. **Result Normalization:**
   - Extract execution metrics (time, memory usage)
   - Convert QuEST output to ExecutionResult format
   - Include backend-specific metadata (rank, precision, GPU usage)

---

### Step 1.4: Handle QuEST-Specific Features

**Precision Configuration:**

1. Detect QuEST build configuration (single/double/quad precision)
2. Add configuration option in Proxima settings: `quest_precision: "double"`
3. Apply precision to qureg creation and gate operations
4. Report precision in capabilities and results

**GPU Acceleration:**

1. Detect CUDA availability during initialization
2. Configure QuEST to use GPU if available: set `CUDA` flag in environment
3. Monitor GPU memory usage during execution
4. Fall back to CPU if GPU initialization fails
5. Report GPU usage in ExecutionResult metadata

**Rank Truncation (for Density Matrices):**

1. Configure truncation threshold in options: `truncation_threshold: 1e-4`
2. Apply SVD-based truncation after noisy gate operations
3. Monitor rank growth throughout execution
4. Report final rank in results metadata

**OpenMP Parallelization:**

1. Detect available CPU cores
2. Configure OpenMP thread count: `setQuESTEnvThreadCount(env, num_threads)`
3. Use parallel mode for state vector operations
4. Monitor parallel efficiency

---

### Step 1.5: Register QuEST in Backend Registry

**File Location:** `src/proxima/backends/registry.py`

**Registration Steps:**

1. Import QuestAdapter class in registry module
2. Add QuEST detection logic in `discover()` method:
   - Try importing pyquest module
   - If successful, instantiate QuestAdapter
   - Add to available backends list
   - If import fails, mark as unavailable with reason
3. Update backend priority list for auto-selection
4. Add QuEST to backend comparison matrix

**Configuration Entry:**

Add to `src/proxima/config/defaults.py`:

```
Backend configuration:
- quest_enabled: true
- quest_default_precision: "double"
- quest_gpu_enabled: auto  # auto/true/false
- quest_truncation_threshold: 1e-4
- quest_max_qubits: 30
```

---

### Step 1.6: Implement Error Handling

**Error Categories:**

1. **Installation Errors:**
   - QuEST library not found
   - Python bindings missing
   - CUDA not available (for GPU mode)
   - Action: Clear error messages, suggest installation steps

2. **Resource Errors:**
   - Insufficient memory for qureg allocation
   - GPU out of memory
   - Action: Estimate required resources, suggest smaller circuit or different backend

3. **Circuit Errors:**
   - Unsupported gate types
   - Invalid qubit indices
   - Action: Provide gate decomposition suggestions or alternative backends

4. **Execution Errors:**
   - QuEST runtime errors
   - Memory corruption
   - Action: Clean up resources, provide diagnostic information

**Error Handling Pattern:**

1. Wrap all QuEST API calls in try-except blocks
2. Map QuEST error codes to Proxima exceptions
3. Ensure proper cleanup on errors (destroy qureg, environment)
4. Log detailed error information for debugging
5. Return user-friendly error messages

---

## Phase 2: cuQuantum Integration

**Duration:** 1-2 weeks  
**Goal:** Enable GPU-accelerated state vector simulations via cuQuantum through Qiskit Aer

---

### Step 2.1: Understand cuQuantum Architecture

**Research Tasks:**

1. Study cuQuantum SDK documentation (https://github.com/NVIDIA/cuQuantum)
2. Focus on cuStateVec library for state vector operations
3. Understand Qiskit Aer's cuQuantum integration:
   - AerSimulator with `device='GPU'` and `method='statevector'`
   - Backend configuration options for cuQuantum
4. Review cuQuantum performance characteristics and limitations
5. Identify GPU memory requirements for different qubit counts

**Key Integration Points:**

- cuQuantum is used through Qiskit Aer, not as standalone backend
- Extend existing `qiskit_adapter.py` rather than creating new adapter
- GPU detection and fallback logic required
- Memory estimation critical for large circuits

---

### Step 2.2: Extend Qiskit Aer Adapter for GPU

**File Location:** `src/proxima/backends/qiskit_adapter.py`

**Modification Strategy:**

1. **Add GPU Detection Method:**
   - Check for CUDA availability using `pycuda` or `cupy`
   - Verify cuQuantum libraries are installed
   - Detect available GPU devices and memory
   - Store GPU availability flag

2. **Extend Capabilities Reporting:**
   - Add `supports_gpu: True/False` to capabilities
   - Report GPU device information (name, memory, compute capability)
   - Indicate cuQuantum availability

3. **Modify Backend Initialization:**
   - Create two backend instances:
     - Standard AerSimulator for CPU execution
     - GPU-enabled AerSimulator with device='GPU' for cuQuantum
   - Configure GPU simulator with cuStateVec method
   - Handle GPU initialization failures gracefully

4. **Implement GPU Execution Path:**
   - Add method: `_execute_on_gpu(circuit, options)`
   - Configure AerSimulator with GPU settings:
     - Set `device='GPU'`
     - Set `method='statevector'`
     - Configure `cuStateVec_enable=True`
     - Set GPU-specific options (blocking, streams)
   - Monitor GPU memory during execution
   - Capture GPU performance metrics

5. **Add Fallback Logic:**
   - Estimate GPU memory requirement: 2^n * 16 bytes (complex128)
   - Compare with available GPU memory
   - If insufficient, fall back to CPU execution
   - Log fallback decision with reason

**GPU Configuration Options:**

```
GPU-specific settings:
- gpu_device_id: 0 (which GPU to use)
- gpu_memory_limit: "auto" (or explicit MB value)
- custatevec_workspace_size: 1GB (scratch memory)
- gpu_blocking: true (wait for completion)
- gpu_streams: 1 (number of CUDA streams)
```

---

### Step 2.3: Create cuQuantum Configuration Helper

**File Location:** `src/proxima/backends/cuquantum_adapter.py`

**Purpose:** Provide convenience wrapper and configuration management for GPU execution

**Class Structure:**

```
CuQuantumAdapter (extends QiskitAdapter)
├── __init__()
│   └── Call parent, configure GPU-specific settings
├── get_name() → "cuquantum"
├── get_capabilities()
│   └── GPU-only capabilities, higher max qubits (35+)
├── validate_circuit(circuit)
│   └── Check GPU memory requirements
├── estimate_resources(circuit)
│   └── GPU memory: 2^n qubits * 16 bytes + workspace
└── execute(circuit, options)
    └── Force GPU execution or error if unavailable
```

**Implementation Details:**

1. **Initialization:**
   - Verify GPU availability (error if not available)
   - Load cuQuantum libraries
   - Configure default GPU settings
   - Initialize GPU device

2. **Circuit Validation:**
   - Calculate state vector memory: `(2^num_qubits) * 16` bytes
   - Add workspace overhead: ~1-2 GB
   - Compare with available GPU memory
   - Reject if insufficient memory

3. **Execution Logic:**
   - Transpile circuit for GPU-optimal gate set
   - Configure AerSimulator for GPU:
     - `AerSimulator(method='statevector', device='GPU')`
     - Enable cuStateVec via backend options
   - Execute on GPU
   - Transfer results from GPU to CPU memory
   - Report GPU metrics (execution time, memory used)

4. **Error Handling:**
   - GPU out of memory → clear error message
   - CUDA driver errors → suggest driver update
   - cuQuantum library not found → installation instructions
   - GPU device busy → retry or fallback logic

---

### Step 2.4: Optimize GPU Memory Management

**Memory Estimation Formula:**

```
Total GPU Memory Required:
= State Vector Size + Workspace + Overhead
= (2^n * 16 bytes) + (1-2 GB) + (500 MB)

Where:
- n = number of qubits
- 16 bytes = sizeof(complex128)
- Workspace = cuStateVec scratch memory
- Overhead = CUDA runtime, driver buffers
```

**Optimization Techniques:**

1. **Pre-Execution Checks:**
   - Query GPU memory before circuit execution
   - Calculate exact memory requirement
   - Reserve memory if available
   - Clear GPU cache if needed

2. **Memory Pooling:**
   - Reuse GPU memory allocations across multiple executions
   - Implement memory pool for repeated simulations
   - Release pool on adapter cleanup

3. **Batch Processing:**
   - For multiple circuits, execute sequentially
   - Clean up between executions to free memory
   - Monitor memory fragmentation

4. **Fallback Strategy:**
   - If GPU memory insufficient, use CPU AerSimulator
   - Log fallback reason
   - Suggest reducing circuit size or using different backend

---

### Step 2.5: Integration Testing

**Test Scenarios:**

1. **GPU Availability Tests:**
   - Test with GPU present
   - Test without GPU (should fall back or error)
   - Test with multiple GPUs (select correct device)

2. **Memory Tests:**
   - Small circuit (10 qubits) → should succeed
   - Large circuit (25 qubits) → verify memory checks
   - Out of memory circuit (30+ qubits) → proper error handling

3. **Performance Tests:**
   - Compare GPU vs CPU execution time
   - Verify GPU speedup for large circuits
   - Monitor GPU utilization

4. **Correctness Tests:**
   - Compare GPU results with CPU results
   - Verify statevector accuracy
   - Test various gate types

---

## Phase 3: qsim Integration

**Duration:** 1-2 weeks  
**Goal:** Enable high-performance CPU state vector simulations using qsim via qsimcirq

---

### Step 3.1: Understand qsim Architecture

**Research Tasks:**

1. Study qsim documentation (https://github.com/quantumlib/qsim)
2. Understand qsimcirq interface:
   - QSimSimulator class
   - QSimhSimulator for hybrid algorithms
   - Integration with Cirq circuits
3. Review qsim performance features:
   - AVX2/AVX512 vectorization
   - OpenMP parallelization
   - Gate fusion optimization
4. Understand qsim limitations and gate support

**Key Features:**

- Highly optimized for Intel/AMD CPUs with AVX2
- Automatic gate fusion for performance
- OpenMP parallelization across CPU cores
- Seamless Cirq integration via qsimcirq

---

### Step 3.2: Create qsim Adapter Leveraging Cirq

**File Location:** `src/proxima/backends/qsim_adapter.py`

**Integration Strategy:**

Since qsim integrates with Cirq, leverage existing `cirq_adapter.py`:

**Class Structure:**

```
QsimAdapter (extends CirqAdapter or create standalone)
├── __init__()
│   └── Initialize qsimcirq.QSimSimulator
├── get_name() → "qsim"
├── get_version() → Query qsim version
├── get_capabilities()
│   └── State vector only, high max qubits (30+), CPU-optimized
├── validate_circuit(circuit)
│   └── Check qsim gate support
├── estimate_resources(circuit)
│   └── Memory: 2^n * 16 bytes (state vector only)
└── execute(circuit, options)
    └── Use QSimSimulator to run circuit
```

**Implementation Details:**

1. **Initialization:**
   - Import qsimcirq module
   - Create QSimSimulator instance
   - Configure qsim options (threads, fusion)
   - Detect CPU features (AVX2, AVX512)

2. **Circuit Handling:**
   - Accept Cirq circuit format (already compatible)
   - Validate all gates are qsim-supported
   - Apply gate fusion hints if available
   - Preserve circuit structure for optimization

3. **Execution Configuration:**
   ```
   qsim_options:
   - t: number of OpenMP threads
   - f: gate fusion strategy
   - v: verbosity level
   - disable_gate_fusion: false (default)
   ```

4. **Simulator Execution:**
   - Convert circuit to qsim-compatible format (usually no conversion needed)
   - Run simulation using `qsim_simulator.simulate(circuit)`
   - For shots-based sampling: use `qsim_simulator.run(circuit, repetitions=shots)`
   - For state vector: use `qsim_simulator.simulate()` and extract final state

5. **Result Processing:**
   - Extract state vector or measurement counts
   - Convert to Proxima's ExecutionResult format
   - Include qsim-specific metadata (gate fusion applied, threads used)

---

### Step 3.3: Optimize qsim Performance

**Configuration Options:**

1. **Thread Control:**
   - Detect CPU core count using psutil
   - Set OpenMP threads: `qsim_options.t = num_cores`
   - Leave 1-2 cores free for system operations
   - Allow user override via configuration

2. **Gate Fusion:**
   - Enable automatic gate fusion (default)
   - Configure fusion strategy:
     - `0`: No fusion
     - `1`: Aggressive fusion (maximum performance)
     - `2`: Balanced fusion
   - Monitor fusion effectiveness in logs

3. **Vectorization:**
   - Verify AVX2/AVX512 support at runtime
   - qsim automatically uses best available instructions
   - No manual configuration needed
   - Report vector instruction set in capabilities

**Performance Tuning:**

1. **For Small Circuits (< 20 qubits):**
   - Use fewer threads (4-8)
   - Aggressive gate fusion
   - Minimize measurement overhead

2. **For Large Circuits (20-30 qubits):**
   - Use all available threads
   - Balanced gate fusion
   - Pre-allocate memory

3. **For Very Large Circuits (30+ qubits):**
   - Consider memory-mapped state vector
   - Sequential execution with checkpointing
   - Warn about memory requirements

---

### Step 3.4: Handle qsim Limitations

**Gate Support:**

qsim supports most common gates but has limitations:

1. **Supported Gates:**
   - Single-qubit: X, Y, Z, H, S, T, Rx, Ry, Rz
   - Two-qubit: CNOT, CZ, SWAP, ISWAP
   - Multi-qubit: CCX (Toffoli), CCZ

2. **Unsupported Features:**
   - Mid-circuit measurements (limited support)
   - Reset operations
   - Classical control flow

**Handling Strategy:**

1. **Circuit Validation:**
   - Check for unsupported gates before execution
   - Decompose complex gates into qsim-supported gates
   - Warn user about limitations

2. **Fallback Logic:**
   - If circuit contains unsupported features:
     - Attempt automatic decomposition
     - If decomposition fails, suggest Cirq backend
     - Log fallback reason

3. **Measurement Handling:**
   - For final measurements: full qsim support
   - For mid-circuit measurements:
     - Check qsim version for support
     - If unsupported, use Cirq simulator
     - Document limitation in results

---

### Step 3.5: Register qsim in Backend System

**Registry Integration:**

1. **Discovery Logic:**
   - Try importing qsimcirq module
   - Check for qsim C++ library
   - Verify CPU supports AVX2 (minimum requirement)
   - Add to available backends if successful

2. **Capability Reporting:**
   ```
   QsimCapabilities:
   - simulator_types: [STATE_VECTOR]
   - max_qubits: 35 (depends on RAM)
   - supports_noise: false
   - supports_gpu: false
   - supports_batching: true
   - custom_features:
     - avx2_enabled: true/false
     - avx512_enabled: true/false
     - gate_fusion_enabled: true
     - max_threads: detected CPU cores
   ```

3. **Priority in Auto-Selection:**
   - High priority for state vector simulations on CPU
   - Lower than cuQuantum if GPU available
   - Higher than standard Cirq for performance-critical tasks

---

## Phase 4: Unified Backend Selection Enhancement

**Duration:** 1 week  
**Goal:** Update intelligent backend selection to consider new backends

---

### Step 4.1: Update Backend Selector Logic

**File Location:** `src/proxima/intelligence/selector.py`

**Enhancement Areas:**

1. **Extended Selection Algorithm:**

```
Circuit Analysis:
├── Qubit count (n)
├── Gate types and depth
├── Simulation type (SV vs DM)
├── Noise model present?
├── GPU available?
└── Performance priority

Backend Ranking:
├── For State Vector + GPU available:
│   ├── First choice: cuQuantum (GPU-accelerated)
│   ├── Second choice: qsim (CPU-optimized)
│   └── Third choice: Cirq/Qiskit Aer (standard)
│
├── For State Vector + CPU only:
│   ├── First choice: qsim (AVX-optimized)
│   ├── Second choice: QuEST (high-performance)
│   └── Third choice: Cirq/Qiskit Aer (standard)
│
├── For Density Matrix:
│   ├── First choice: QuEST (supports DM, GPU option)
│   ├── Second choice: Cirq DensityMatrixSimulator
│   └── Third choice: Qiskit Aer
│
└── For Noisy Circuits:
    ├── First choice: QuEST (noise support + DM)
    ├── Second choice: Qiskit Aer with noise
    └── Third choice: Cirq with noise
```

2. **GPU-Aware Selection:**
   - Detect GPU availability at selection time
   - If GPU present and circuit suitable (SV, 20+ qubits):
     - Prefer cuQuantum
     - Estimate GPU memory requirement
     - Fall back if insufficient GPU memory
   - If no GPU, prefer CPU-optimized backends (qsim)

3. **Memory-Based Selection:**
   - Calculate memory requirements for each backend:
     - State Vector: 2^n * 16 bytes
     - Density Matrix: 2^(2n) * 16 bytes
   - Compare with available RAM/GPU memory
   - Eliminate backends with insufficient resources
   - Rank by memory efficiency

4. **Performance-Based Selection:**
   - Maintain performance history database:
     - Backend name, circuit size, execution time
     - Store in SQLite or JSON file
   - Query history for similar circuit sizes
   - Rank backends by historical performance
   - Update after each execution

---

### Step 4.2: Update Backend Comparison Matrix

**File Location:** `src/proxima/data/compare.py`

**Comparison Dimensions:**

| Backend   | SV | DM | GPU | CPU Opt | Noise | Max Qubits | Use Case                    |
|-----------|----|----|-----|---------|-------|------------|-----------------------------|
| LRET      | ✓  | ✓  | ✗   | ✗       | ✓     | 15         | Custom rank-reduction       |
| Cirq      | ✓  | ✓  | ✗   | ✗       | ✓     | 20         | General-purpose             |
| Qiskit Aer| ✓  | ✓  | ✗   | ✗       | ✓     | 30         | Qiskit ecosystem            |
| QuEST     | ✓  | ✓  | ✓   | ✓       | ✓     | 30         | High-performance research   |
| cuQuantum | ✓  | ✗  | ✓   | ✗       | ✗     | 35+        | GPU-accelerated large SV    |
| qsim      | ✓  | ✗  | ✗   | ✓✓      | ✗     | 35+        | CPU-optimized large SV      |

**Comparison Metrics:**

1. **Performance Comparison:**
   - Execution time for same circuit
   - Scaling with qubit count
   - Parallel efficiency

2. **Resource Comparison:**
   - Peak memory usage
   - GPU utilization (if applicable)
   - CPU utilization

3. **Accuracy Comparison:**
   - State vector fidelity
   - Measurement distribution similarity
   - Numerical stability

---

### Step 4.3: Configuration Updates

**File Location:** `src/proxima/config/defaults.py`

**New Configuration Options:**

```yaml
backends:
  auto_selection:
    prefer_gpu: true  # Prefer GPU backends if available
    prefer_performance: true  # Prioritize speed over memory
    min_qubits_for_gpu: 20  # Only use GPU for circuits >= this size
    
  quest:
    enabled: true
    precision: double  # single, double, quad
    gpu_enabled: auto  # auto, true, false
    truncation_threshold: 1e-4
    
  cuquantum:
    enabled: auto  # auto (if GPU present), true, false
    gpu_device: 0  # Which GPU to use
    memory_limit: auto  # or explicit MB value
    fallback_to_cpu: true
    
  qsim:
    enabled: true
    threads: auto  # or explicit number
    gate_fusion: aggressive  # none, balanced, aggressive
    
backend_priorities:
  state_vector_gpu: [cuquantum, quest, qsim, cirq]
  state_vector_cpu: [qsim, quest, cirq, qiskit]
  density_matrix: [quest, cirq, qiskit]
  noisy_circuit: [quest, qiskit, cirq]
```

---

## Phase 5: Testing & Validation

**Duration:** 2 weeks  
**Goal:** Comprehensive testing of all new backends

---

### Step 5.1: Unit Testing

**Test File Structure:**

```
tests/backends/
├── test_quest_adapter.py
├── test_cuquantum_adapter.py
├── test_qsim_adapter.py
└── test_backend_selection.py
```

**Test Cases for Each Backend:**

1. **Initialization Tests:**
   - Backend instantiation succeeds
   - Capabilities correctly reported
   - Version information retrieved
   - Resource detection works

2. **Circuit Translation Tests:**
   - Basic gates translate correctly
   - Parameterized gates work
   - Multi-qubit gates handled
   - Invalid gates rejected

3. **Execution Tests:**
   - Simple circuits execute
   - Results are accurate
   - Memory management works
   - Cleanup is proper

4. **Error Handling Tests:**
   - Missing dependencies handled
   - Resource exhaustion handled
   - Invalid circuits rejected
   - Error messages are clear

---

### Step 5.2: Integration Testing

**Test Scenarios:**

1. **Backend Registry Tests:**
   - All backends discovered correctly
   - Unavailable backends marked properly
   - Selection logic works with new backends
   - Priority ordering correct

2. **Multi-Backend Comparison Tests:**
   - Same circuit on all backends
   - Results agree within tolerance
   - Performance metrics collected
   - Comparison report generated

3. **Fallback Logic Tests:**
   - GPU unavailable → falls back to CPU
   - Memory insufficient → suggests smaller backend
   - Unsupported feature → recommends alternative

4. **Configuration Tests:**
   - All configuration options work
   - Defaults are sensible
   - Overrides apply correctly
   - Invalid configs rejected

---

### Step 5.3: Performance Benchmarking

**Benchmark Suite:**

1. **Small Circuits (5-10 qubits):**
   - Measure overhead of each backend
   - Compare initialization time
   - Validate all backends produce correct results

2. **Medium Circuits (15-20 qubits):**
   - Measure execution time
   - Monitor memory usage
   - Compare performance across backends

3. **Large Circuits (25-30 qubits):**
   - GPU vs CPU comparison
   - Memory scaling analysis
   - Identify performance bottlenecks

4. **Noise and Density Matrix:**
   - Test QuEST with various noise models
   - Compare DM vs SV performance
   - Validate noise fidelity

**Benchmark Metrics:**

- Execution time (wall clock)
- Peak memory usage (RSS)
- CPU utilization percentage
- GPU utilization (if applicable)
- Result accuracy (fidelity)

---

### Step 5.4: Validation Against Known Results

**Validation Strategy:**

1. **Standard Test Circuits:**
   - Bell state preparation
   - GHZ state preparation
   - Quantum Fourier Transform
   - Grover's algorithm
   - Variational Quantum Eigensolver

2. **Validation Method:**
   - Run on reference backend (Qiskit Aer CPU)
   - Run on each new backend
   - Compare state vectors or measurement distributions
   - Assert fidelity > 0.9999 or distribution similarity > 0.999

3. **Edge Cases:**
   - Maximum qubit count for each backend
   - Minimum resource circuits
   - All supported gate types
   - Parameterized circuits

---

## Phase 6: Documentation & Deployment

**Duration:** 1 week  
**Goal:** Complete documentation and deployment preparation

---

### Step 6.1: User Documentation

**Documentation Files to Create/Update:**

1. **Installation Guide:**
   - `docs/backends/quest-installation.md`
     - Building from source instructions
     - Pre-built binaries
     - GPU setup (CUDA, cuQuantum)
     - Verification steps
   
   - `docs/backends/cuquantum-installation.md`
     - NVIDIA driver requirements
     - CUDA Toolkit installation
     - cuQuantum SDK setup
     - GPU verification
   
   - `docs/backends/qsim-installation.md`
     - Python package installation
     - CPU feature verification (AVX2)
     - Configuration options

2. **Usage Guide:**
   - `docs/backends/quest-usage.md`
     - Basic examples
     - Precision configuration
     - GPU vs CPU selection
     - Density matrix mode
     - Performance tuning
   
   - `docs/backends/cuquantum-usage.md`
     - GPU selection
     - Memory management
     - Performance optimization
     - Troubleshooting
   
   - `docs/backends/qsim-usage.md`
     - Threading configuration
     - Gate fusion options
     - Performance tips

3. **Backend Comparison Guide:**
   - `docs/backends/backend-selection.md`
     - Decision flowchart
     - Performance characteristics
     - Use case recommendations
     - Resource requirements table

---

### Step 6.2: API Documentation

**Update API Reference:**

1. **Backend Adapter Classes:**
   - Document QuestAdapter class and methods
   - Document CuQuantumAdapter class and methods
   - Document QsimAdapter class and methods
   - Include code examples for each

2. **Configuration Schema:**
   - Document all new configuration options
   - Provide default values and valid ranges
   - Explain impact of each option

3. **Result Format:**
   - Document backend-specific metadata fields
   - Explain performance metrics
   - Show example ExecutionResult objects

---

### Step 6.3: Update CLI Help and Examples

**CLI Documentation:**

1. **Update `proxima backends list` output:**
   - Show all 6 backends (LRET, Cirq, Qiskit, QuEST, cuQuantum, qsim)
   - Display capabilities for each
   - Show availability status

2. **Update `proxima backends info` command:**
   - Detailed information for new backends
   - Installation status
   - Hardware requirements
   - Example commands

3. **Add Backend-Specific Examples:**
   ```bash
   # QuEST examples
   proxima run --backend quest --simulator density-matrix "noisy circuit"
   proxima run --backend quest --gpu "large statevector"
   
   # cuQuantum examples
   proxima run --backend cuquantum "30-qubit circuit"
   proxima run --backend cuquantum --gpu-device 1 "multi-gpu simulation"
   
   # qsim examples
   proxima run --backend qsim --threads 16 "cpu-optimized circuit"
   proxima run --backend qsim --gate-fusion aggressive "fusion test"
   ```

---

### Step 6.4: Create Migration Guide

**File:** `docs/migration/adding-backends-guide.md`

**Content:**

1. **For Existing Proxima Users:**
   - What changed in backend system
   - How to install new backends
   - Configuration file updates needed
   - Breaking changes (if any)

2. **Upgrade Instructions:**
   - Update Proxima to version X.Y.Z
   - Install optional backend dependencies
   - Update configuration files
   - Test backend availability

3. **Troubleshooting Common Issues:**
   - Backend not detected
   - GPU not working with cuQuantum
   - qsim performance not optimal
   - QuEST build failures

---

### Step 6.5: Update Integration Tests in CI/CD

**CI/CD Configuration Updates:**

1. **GitHub Actions Workflows:**
   - Add backend installation steps to CI pipeline
   - Install QuEST (CPU-only for CI)
   - Install qsimcirq
   - Skip cuQuantum tests (no GPU in CI) or use mock
   
2. **Test Matrix:**
   ```yaml
   strategy:
     matrix:
       backend: [cirq, qiskit, quest, qsim]
       python-version: [3.11, 3.12]
   ```

3. **Mock Tests for GPU:**
   - Create mock GPU detection for cuQuantum tests
   - Test fallback logic without actual GPU
   - Document GPU testing requirements

4. **Performance Regression Tests:**
   - Benchmark new backends on each commit
   - Compare against baseline performance
   - Alert on significant regressions

---

### Step 6.6: Deployment Checklist

**Pre-Release Validation:**

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Backend auto-selection works correctly
- [ ] Documentation is complete and accurate
- [ ] Configuration schema validated
- [ ] Performance benchmarks meet targets
- [ ] Error messages are user-friendly
- [ ] Installation guides verified on clean systems
- [ ] GPU tests pass on NVIDIA hardware
- [ ] CPU tests pass on various architectures

**Release Artifacts:**

- [ ] Updated PyPI package with new dependencies
- [ ] Docker image with all backends (CPU-only)
- [ ] Separate Docker image with GPU support
- [ ] Binary releases for Windows/Mac/Linux
- [ ] Documentation website updated
- [ ] Migration guide published
- [ ] Release notes written

**Post-Release:**

- [ ] Monitor GitHub issues for backend-specific problems
- [ ] Update backend compatibility matrix based on user feedback
- [ ] Collect performance benchmarks from users
- [ ] Iterate on documentation based on questions

---

## Summary

This guide provides a complete implementation roadmap for integrating QuEST, cuQuantum, and qsim into Proxima. The implementation follows a phased approach:

1. **Phase 1 (QuEST):** Most complex, requires C++ interop, supports both SV and DM
2. **Phase 2 (cuQuantum):** Extends existing Qiskit adapter, GPU-focused
3. **Phase 3 (qsim):** Leverages Cirq, CPU-optimized, simplest integration
4. **Phase 4:** Updates selection logic to utilize all backends optimally
5. **Phase 5:** Comprehensive testing ensures reliability
6. **Phase 6:** Documentation and deployment preparation

**Key Success Factors:**

- Maintain existing Proxima architecture and interfaces
- Ensure graceful fallbacks when backends unavailable
- Provide clear error messages and installation guidance
- Optimize for performance on each backend's target hardware
- Comprehensive testing across different hardware configurations
- Excellent documentation for users and developers

**Expected Timeline:** 7-9 weeks total for complete implementation and testing.

**Next Steps for Implementation:**

1. Start with Phase 1 (QuEST) as it's the foundation for understanding low-level integration
2. Proceed to Phase 3 (qsim) for quick win with Cirq integration
3. Complete Phase 2 (cuQuantum) leveraging Qiskit knowledge
4. Enhance selection logic in Phase 4
5. Thorough testing in Phase 5
6. Documentation and deployment in Phase 6
