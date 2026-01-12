# Proxima CLI Reference

> **Version:** 1.0  
> **Last Updated:** January 12, 2026

---

## Overview

The Proxima CLI provides commands for running quantum circuit simulations, managing backends, comparing results, and configuring the system.

---

## Global Options

```bash
proxima [OPTIONS] COMMAND [ARGS]

Options:
  --version           Show version and exit
  --verbose, -v       Enable verbose output (can be repeated: -vv, -vvv)
  --quiet, -q         Suppress non-essential output
  --config FILE       Path to configuration file
  --no-color          Disable colored output
  --help              Show help message and exit
```

---

## Commands

### `proxima run`

Execute a quantum circuit simulation.

```bash
proxima run [OPTIONS] CIRCUIT_FILE

Arguments:
  CIRCUIT_FILE        Path to circuit file (.py, .qasm, .json)

Options:
  --backend, -b TEXT  Backend to use (auto, quest, cuquantum, qsim, cirq, 
                      qiskit, lret) [default: auto]
  --sim-type TEXT     Simulation type (statevector, density_matrix)
                      [default: statevector]
  --shots, -s INT     Number of measurement shots [default: 1024]
  --seed INT          Random seed for reproducibility
  --gpu               Enable GPU acceleration
  --threads INT       Number of CPU threads to use
  --output, -o FILE   Output file for results (json, csv, xlsx)
  --profile           Enable performance profiling
  --dry-run           Validate without execution
  --help              Show help message and exit
```

**Examples:**

```bash
# Basic execution with auto-selected backend
proxima run my_circuit.py

# Use QuEST backend with 10000 shots
proxima run --backend quest --shots 10000 my_circuit.py

# GPU-accelerated execution with cuQuantum
proxima run --backend cuquantum --gpu large_circuit.py

# Density matrix simulation
proxima run --backend quest --sim-type density_matrix noisy_circuit.py

# Save results to file
proxima run --backend qsim --output results.json my_circuit.py

# Dry run to validate circuit
proxima run --dry-run --backend quest my_circuit.py

# Verbose output with profiling
proxima run -vv --profile --backend qsim my_circuit.py
```

---

### `proxima backends`

Manage and inspect quantum simulation backends.

#### `proxima backends list`

List all available backends.

```bash
proxima backends list [OPTIONS]

Options:
  --all, -a           Show all backends (including unavailable)
  --json              Output as JSON
  --help              Show help message and exit
```

**Example Output:**

```
┌─────────────┬───────────┬────────────────────────┬───────────┐
│ Backend     │ Status    │ Capabilities           │ Max Qubits│
├─────────────┼───────────┼────────────────────────┼───────────┤
│ quest       │ ✓ Ready   │ SV, DM, GPU, Noise     │ 30        │
│ cuquantum   │ ✓ Ready   │ SV, GPU                │ 35        │
│ qsim        │ ✓ Ready   │ SV, AVX512             │ 35        │
│ cirq        │ ✓ Ready   │ SV, DM, Noise          │ 20        │
│ qiskit      │ ✓ Ready   │ SV, DM, Noise          │ 30        │
│ lret        │ ✓ Ready   │ SV, DM, Low-Rank       │ 15        │
└─────────────┴───────────┴────────────────────────┴───────────┘

Legend: SV=State Vector, DM=Density Matrix
```

---

#### `proxima backends info`

Get detailed information about a specific backend.

```bash
proxima backends info [OPTIONS] BACKEND_NAME

Arguments:
  BACKEND_NAME        Name of the backend (quest, cuquantum, qsim, etc.)

Options:
  --json              Output as JSON
  --verbose, -v       Show additional details
  --help              Show help message and exit
```

**Example Output:**

```bash
$ proxima backends info quest

QuEST Backend
═════════════

Version:        3.5.0
Status:         Available ✓
Library Path:   /usr/local/lib/libQuEST.so

Capabilities:
  ├─ State Vector:      ✓
  ├─ Density Matrix:    ✓
  ├─ GPU Acceleration:  ✓ (CUDA 12.0)
  ├─ Noise Simulation:  ✓
  ├─ Gate Fusion:       ✓
  └─ Max Qubits:        30

Hardware:
  ├─ CPU Threads:       16 (available)
  ├─ GPU:               NVIDIA RTX 4090
  └─ GPU Memory:        24576 MB

Configuration:
  ├─ Default Precision: double
  ├─ GPU Enabled:       auto
  └─ Thread Count:      auto

Supported Gates:
  Single-qubit: X, Y, Z, H, S, T, Rx, Ry, Rz, U3
  Two-qubit:    CNOT, CZ, CRx, CRy, CRz, SWAP
  Multi-qubit:  CCX, CCZ, Multi-controlled

Example:
  proxima run --backend quest --gpu my_circuit.py
```

---

#### `proxima backends compare`

Compare multiple backends.

```bash
proxima backends compare [OPTIONS] BACKENDS...

Arguments:
  BACKENDS            Backend names to compare (space-separated)

Options:
  --output, -o FILE   Output comparison to file
  --json              Output as JSON
  --help              Show help message and exit
```

**Example:**

```bash
$ proxima backends compare quest cuquantum qsim

Backend Comparison
══════════════════

                    │ quest      │ cuquantum  │ qsim       │
────────────────────┼────────────┼────────────┼────────────┤
State Vector        │ ✓          │ ✓          │ ✓          │
Density Matrix      │ ✓          │ ✗          │ ✗          │
GPU Support         │ ✓          │ ✓          │ ✗          │
CPU Optimized       │ ✓          │ ✗          │ ✓✓         │
Noise Simulation    │ ✓          │ ✗          │ ✗          │
Max Qubits          │ 30         │ 35+        │ 35+        │
────────────────────┼────────────┼────────────┼────────────┤
Best For            │ Research   │ Large GPU  │ Large CPU  │
```

---

#### `proxima backends test`

Test backend functionality.

```bash
proxima backends test [OPTIONS] [BACKEND_NAME]

Arguments:
  BACKEND_NAME        Backend to test (optional, tests all if omitted)

Options:
  --all               Test all available backends
  --quick             Run quick smoke tests only
  --verbose, -v       Show detailed test output
  --help              Show help message and exit
```

**Example:**

```bash
$ proxima backends test --all

Testing Backends
════════════════

quest:
  ├─ Import:          ✓ (0.12s)
  ├─ Initialization:  ✓ (0.05s)
  ├─ Bell State:      ✓ (0.03s)
  ├─ 10-Qubit GHZ:    ✓ (0.15s)
  └─ Cleanup:         ✓ (0.01s)
  Status: PASSED

cuquantum:
  ├─ Import:          ✓ (0.08s)
  ├─ GPU Detection:   ✓ (NVIDIA RTX 4090)
  ├─ Bell State:      ✓ (0.02s)
  ├─ 20-Qubit GHZ:    ✓ (0.45s)
  └─ Cleanup:         ✓ (0.01s)
  Status: PASSED

qsim:
  ├─ Import:          ✓ (0.10s)
  ├─ AVX Detection:   ✓ (AVX512)
  ├─ Bell State:      ✓ (0.01s)
  ├─ 20-Qubit GHZ:    ✓ (0.25s)
  └─ Cleanup:         ✓ (0.00s)
  Status: PASSED

Summary: 3/3 backends passed
```

---

### `proxima compare`

Run the same circuit on multiple backends and compare results.

```bash
proxima compare [OPTIONS] CIRCUIT_FILE

Arguments:
  CIRCUIT_FILE        Path to circuit file

Options:
  --backends, -b TEXT Backend names (comma-separated or 'all')
                      [default: auto]
  --shots, -s INT     Number of shots per backend [default: 1024]
  --output, -o FILE   Output file for comparison report
  --format TEXT       Output format (text, json, csv, xlsx) [default: text]
  --fidelity          Calculate cross-backend fidelity
  --help              Show help message and exit
```

**Example:**

```bash
$ proxima compare --backends quest,cuquantum,qsim --shots 10000 bell_state.py

Multi-Backend Comparison Report
═══════════════════════════════

Circuit: bell_state.py (2 qubits, depth 2)
Shots: 10000 per backend

Results:
┌───────────┬───────────┬───────────┬──────────────┬────────────┐
│ Backend   │ |00⟩      │ |11⟩      │ Time (ms)    │ Memory (MB)│
├───────────┼───────────┼───────────┼──────────────┼────────────┤
│ quest     │ 4987      │ 5013      │ 12.3         │ 0.02       │
│ cuquantum │ 5021      │ 4979      │ 8.1          │ 256.0      │
│ qsim      │ 4995      │ 5005      │ 5.2          │ 0.02       │
└───────────┴───────────┴───────────┴──────────────┴────────────┘

Cross-Backend Fidelity:
  quest ↔ cuquantum:    0.9999
  quest ↔ qsim:         0.9999
  cuquantum ↔ qsim:     0.9999

Performance Winner: qsim (5.2 ms)
Result Agreement: EXCELLENT (fidelity > 0.999)
```

---

### `proxima config`

Manage Proxima configuration.

#### `proxima config show`

Show current configuration.

```bash
proxima config show [OPTIONS]

Options:
  --section TEXT      Show only specific section
  --defaults          Include default values
  --json              Output as JSON
  --help              Show help message and exit
```

#### `proxima config set`

Set a configuration value.

```bash
proxima config set KEY VALUE

Arguments:
  KEY                 Configuration key (e.g., backends.quest.gpu_enabled)
  VALUE               Value to set
```

#### `proxima config init`

Initialize a new configuration file.

```bash
proxima config init [OPTIONS]

Options:
  --path FILE         Path for config file [default: ./proxima.yaml]
  --force             Overwrite existing file
  --help              Show help message and exit
```

---

### `proxima version`

Show version and system information.

```bash
$ proxima version

Proxima v1.0.0
══════════════

Python:       3.11.5
Platform:     Windows 11 (AMD64)
Install Path: C:\Users\user\envs\proxima

Backends:
  ├─ quest:      3.5.0 ✓
  ├─ cuquantum:  1.2.0 ✓
  ├─ qsim:       0.22.0 ✓
  ├─ cirq:       1.3.0 ✓
  ├─ qiskit:     1.0.0 ✓
  └─ lret:       0.1.0 ✓

Hardware:
  ├─ CPU:        AMD Ryzen 9 5900X (24 threads)
  ├─ RAM:        64 GB
  ├─ GPU:        NVIDIA RTX 4090 (24 GB)
  └─ CUDA:       12.0
```

---

## Backend-Specific Examples

### QuEST Examples

```bash
# Basic QuEST execution
proxima run --backend quest my_circuit.py

# QuEST with GPU acceleration
proxima run --backend quest --gpu my_circuit.py

# QuEST density matrix simulation
proxima run --backend quest --sim-type density_matrix noisy_circuit.py

# QuEST with specific precision
proxima run --backend quest --config quest.precision=double my_circuit.py

# QuEST with custom thread count
proxima run --backend quest --threads 16 my_circuit.py

# QuEST verbose execution with profiling
proxima run --backend quest -vv --profile my_circuit.py
```

### cuQuantum Examples

```bash
# Basic cuQuantum execution (requires NVIDIA GPU)
proxima run --backend cuquantum my_circuit.py

# cuQuantum with specific GPU device
proxima run --backend cuquantum --config cuquantum.device_id=0 my_circuit.py

# cuQuantum for large circuits
proxima run --backend cuquantum --shots 100000 large_circuit.py

# cuQuantum with memory profiling
proxima run --backend cuquantum --profile my_circuit.py
```

### qsim Examples

```bash
# Basic qsim execution (CPU-optimized)
proxima run --backend qsim my_circuit.py

# qsim with all CPU threads
proxima run --backend qsim --threads 0 my_circuit.py  # 0 = all threads

# qsim with gate fusion disabled
proxima run --backend qsim --config qsim.enable_gate_fusion=false my_circuit.py

# qsim for Cirq circuits
proxima run --backend qsim cirq_circuit.py
```

### Multi-Backend Workflow

```bash
# Step 1: Validate circuit on all backends
proxima run --dry-run --backend quest my_circuit.py
proxima run --dry-run --backend cuquantum my_circuit.py
proxima run --dry-run --backend qsim my_circuit.py

# Step 2: Run comparison
proxima compare --backends quest,cuquantum,qsim --shots 10000 my_circuit.py

# Step 3: Run with best backend
proxima run --backend qsim --shots 100000 --output results.json my_circuit.py
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROXIMA_CONFIG` | Path to config file | `~/.proxima/config.yaml` |
| `PROXIMA_LOG_LEVEL` | Logging level | `INFO` |
| `PROXIMA_BACKEND` | Default backend | `auto` |
| `PROXIMA_NO_COLOR` | Disable colors | `false` |
| `PROXIMA_GPU` | Enable GPU by default | `auto` |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Backend not available |
| 4 | Circuit validation failed |
| 5 | Execution error |
| 6 | Resource error (memory, GPU) |
| 7 | Configuration error |

---

## See Also

- [Backend Selection Guide](../backends/backend-selection.md)
- [Configuration Reference](configuration.md)
- [API Reference](../api-reference/)
