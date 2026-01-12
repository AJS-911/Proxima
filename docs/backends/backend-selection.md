# Backend Selection Guide

> **Version:** 1.0  
> **Last Updated:** January 12, 2026  
> **Purpose:** Help users choose the optimal backend for their quantum simulations

---

## Overview

Proxima supports 6 quantum simulation backends, each optimized for different use cases. This guide helps you choose the right backend for your needs.

---

## Backend Comparison Matrix

| Backend | State Vector | Density Matrix | GPU | CPU Optimized | Noise | Max Qubits | Best For |
|---------|:------------:|:--------------:|:---:|:-------------:|:-----:|:----------:|----------|
| **LRET** | ✓ | ✓ | ✗ | ✗ | ✓ | 15 | Custom rank-reduction research |
| **Cirq** | ✓ | ✓ | ✗ | ✗ | ✓ | 20 | General-purpose, Google ecosystem |
| **Qiskit Aer** | ✓ | ✓ | ✗ | ✗ | ✓ | 30 | IBM ecosystem, QASM circuits |
| **QuEST** | ✓ | ✓ | ✓ | ✓ | ✓ | 30 | High-performance research |
| **cuQuantum** | ✓ | ✗ | ✓✓ | ✗ | ✗ | 35+ | GPU-accelerated large circuits |
| **qsim** | ✓ | ✗ | ✗ | ✓✓ | ✗ | 35+ | CPU-optimized large circuits |

---

## Quick Selection Guide

### Decision Tree

```
Start Here
    │
    ├── Need density matrix simulation?
    │   │
    │   ├── YES → Need GPU?
    │   │         ├── YES → QuEST (with GPU)
    │   │         └── NO → QuEST or Cirq
    │   │
    │   └── NO → Continue...
    │
    ├── Need noise simulation?
    │   │
    │   ├── YES → QuEST, Qiskit Aer, or Cirq
    │   │
    │   └── NO → Continue...
    │
    ├── Have NVIDIA GPU?
    │   │
    │   ├── YES → Large circuit (>25 qubits)?
    │   │         ├── YES → cuQuantum
    │   │         └── NO → cuQuantum or QuEST
    │   │
    │   └── NO → Continue...
    │
    ├── Need maximum CPU performance?
    │   │
    │   ├── YES → qsim (AVX2/AVX512)
    │   │
    │   └── NO → Continue...
    │
    └── General purpose → Cirq or Qiskit Aer
```

### One-Line Recommendations

| Scenario | Recommended Backend |
|----------|---------------------|
| Just getting started | `cirq` |
| NVIDIA GPU available | `cuquantum` |
| Maximum CPU performance | `qsim` |
| Noisy quantum circuits | `quest` or `qiskit` |
| Density matrix required | `quest` |
| Research with rank-reduction | `lret` |
| Google/Cirq ecosystem | `qsim` |
| IBM/Qiskit ecosystem | `qiskit` |

---

## Automatic Backend Selection

Proxima can automatically select the optimal backend based on your circuit and hardware:

### Enable Auto-Selection

```bash
# CLI: Use auto mode (default)
proxima run --backend auto my_circuit.py

# CLI: See which backend was selected
proxima run --backend auto --verbose my_circuit.py
```

```python
# Python API
from proxima.intelligence import BackendSelector

selector = BackendSelector()
recommended = selector.select(circuit)

print(f"Recommended: {recommended.backend}")
print(f"Reason: {recommended.reason}")
```

### Auto-Selection Algorithm

The selector considers:

1. **Circuit Properties**
   - Number of qubits
   - Gate types used
   - Circuit depth
   - Measurement requirements

2. **Hardware Capabilities**
   - GPU availability (NVIDIA)
   - CPU features (AVX2, AVX512)
   - Available RAM
   - Number of CPU cores

3. **Simulation Requirements**
   - State vector vs density matrix
   - Noise model presence
   - Shot count
   - Precision requirements

4. **Historical Performance**
   - Past execution times
   - Success rates
   - Resource efficiency

---

## Backend Deep Dive

### LRET (Low-Rank Evolution Tracker)

**Best for:** Research on rank-reduction techniques for density matrix simulation

**Strengths:**
- Custom low-rank approximation
- Memory efficient for specific circuit types
- Research-oriented features

**Limitations:**
- Limited to ~15 qubits
- Specialized use case
- Not optimized for general circuits

**Use when:**
- Researching novel simulation techniques
- Working with circuits amenable to low-rank approximation
- Need custom rank-reduction control

```bash
proxima run --backend lret --max-rank 64 my_circuit.py
```

### Cirq

**Best for:** General-purpose simulation, Google ecosystem, learning

**Strengths:**
- Well-documented, beginner-friendly
- Supports state vector and density matrix
- Good noise model support
- Active community

**Limitations:**
- Not performance-optimized
- Limited to ~20 qubits efficiently
- No GPU acceleration

**Use when:**
- Learning quantum computing
- Developing circuits for Google hardware
- Need a reliable, well-tested simulator

```bash
proxima run --backend cirq my_circuit.py
```

### Qiskit Aer

**Best for:** IBM ecosystem, QASM circuits, comprehensive noise models

**Strengths:**
- Extensive noise modeling
- QASM circuit support
- Good documentation
- Transpiler integration

**Limitations:**
- Can be slower than specialized backends
- GPU support requires separate package

**Use when:**
- Working with IBM Quantum hardware
- Need comprehensive noise simulation
- Using OpenQASM circuits

```bash
proxima run --backend qiskit my_circuit.py
```

### QuEST

**Best for:** High-performance research, density matrix with GPU

**Strengths:**
- Both state vector and density matrix
- Optional GPU acceleration
- OpenMP parallelization
- High precision modes
- Excellent for research

**Limitations:**
- Requires separate installation
- GPU support needs CUDA build

**Use when:**
- Need both simulation types
- Want GPU-accelerated density matrix
- Doing performance-critical research
- Need multi-precision support

```bash
# State vector on GPU
proxima run --backend quest --gpu my_circuit.py

# Density matrix
proxima run --backend quest --sim-type density_matrix my_circuit.py
```

### cuQuantum

**Best for:** GPU-accelerated large state vector simulations

**Strengths:**
- Best GPU performance via NVIDIA cuStateVec
- Handles 35+ qubits with sufficient GPU RAM
- Seamless Qiskit integration

**Limitations:**
- Requires NVIDIA GPU (Volta or newer)
- No density matrix support
- No noise simulation
- GPU memory limits circuit size

**Use when:**
- Have NVIDIA GPU
- Running large state vector circuits
- Need fastest possible execution
- Circuit is noise-free

```bash
proxima run --backend cuquantum my_large_circuit.py
```

### qsim

**Best for:** CPU-optimized large state vector simulations

**Strengths:**
- Best CPU performance (AVX2/AVX512)
- Handles 35+ qubits
- Automatic gate fusion
- OpenMP parallelization
- Cirq-native integration

**Limitations:**
- No density matrix support
- No noise simulation
- CPU-only

**Use when:**
- No GPU available but need performance
- Running large circuits on CPU
- Using Cirq/Google ecosystem
- Need gate fusion optimization

```bash
proxima run --backend qsim --threads 16 my_circuit.py
```

---

## Performance Comparison

### Execution Time (relative, lower is better)

| Qubits | Cirq | Qiskit | QuEST | qsim | cuQuantum |
|--------|------|--------|-------|------|-----------|
| 10 | 1.0x | 1.1x | 0.8x | 0.6x | 0.5x |
| 15 | 1.0x | 1.2x | 0.7x | 0.5x | 0.3x |
| 20 | 1.0x | 1.3x | 0.6x | 0.4x | 0.2x |
| 25 | 1.0x | 1.4x | 0.5x | 0.3x | 0.1x |
| 30 | N/A | 1.0x | 0.4x | 0.2x | 0.05x |

*Benchmarks are approximate and depend on circuit type and hardware*

### Memory Usage

| Qubits | State Vector | Density Matrix |
|--------|-------------|----------------|
| 10 | 16 KB | 16 MB |
| 15 | 512 KB | 512 MB |
| 20 | 16 MB | 16 GB |
| 25 | 512 MB | N/A |
| 30 | 16 GB | N/A |
| 35 | 512 GB | N/A |

---

## Configuration

### Backend Priority Lists

Configure default selection priorities in `proxima.yaml`:

```yaml
backends:
  auto_selection:
    enabled: true
    explain_selection: true
    
  priorities:
    # When GPU is available
    state_vector_gpu:
      - cuquantum
      - quest
      - qsim
      - cirq
      
    # CPU-only
    state_vector_cpu:
      - qsim
      - quest
      - cirq
      - qiskit
      
    # Density matrix
    density_matrix:
      - quest
      - cirq
      - qiskit
      
    # Noisy circuits
    noisy_circuit:
      - quest
      - qiskit
      - cirq
```

### Backend-Specific Settings

```yaml
backends:
  quest:
    enabled: true
    gpu_enabled: auto
    default_precision: double
    max_qubits: 30
    
  cuquantum:
    enabled: true
    gpu_device_id: 0
    memory_limit: auto
    
  qsim:
    enabled: true
    num_threads: auto
    enable_gate_fusion: true
    
  cirq:
    enabled: true
    max_qubits: 20
    
  qiskit:
    enabled: true
    optimization_level: 2
    
  lret:
    enabled: true
    max_rank: 64
```

---

## Checking Backend Availability

### CLI Commands

```bash
# List all backends with status
proxima backends list

# Detailed info about a specific backend
proxima backends info quest

# Compare backend capabilities
proxima backends compare quest cuquantum qsim

# Test all backends
proxima backends test --all
```

### Python API

```python
from proxima.backends import BackendRegistry

# Get all available backends
available = BackendRegistry.discover()
for name, adapter in available.items():
    caps = adapter.get_capabilities()
    print(f"{name}: {caps.max_qubits} qubits, GPU={caps.supports_gpu}")

# Check specific backend
if BackendRegistry.is_available("cuquantum"):
    adapter = BackendRegistry.get("cuquantum")
    print(f"cuQuantum GPU memory: {adapter.get_gpu_memory_mb()} MB")
```

---

## Fallback Logic

Proxima automatically handles backend failures with fallback:

```
Primary Backend Failed
    │
    ├── Reason: GPU out of memory
    │   └── Fallback: Same backend on CPU
    │
    ├── Reason: Backend unavailable
    │   └── Fallback: Next in priority list
    │
    ├── Reason: Unsupported gate
    │   └── Fallback: Decompose and retry, or use different backend
    │
    └── Reason: Resource limit exceeded
        └── Fallback: Lower-capability backend or error
```

### Configuring Fallback

```yaml
backends:
  fallback:
    enabled: true
    max_attempts: 3
    fallback_chain:
      cuquantum: [quest, qsim, cirq]
      quest: [qsim, cirq, qiskit]
      qsim: [quest, cirq, qiskit]
```

---

## Multi-Backend Comparison

Run the same circuit on multiple backends to compare results:

```bash
# Compare backends via CLI
proxima compare --backends quest,cuquantum,qsim my_circuit.py

# Generate comparison report
proxima compare --backends all --output comparison_report.xlsx my_circuit.py
```

```python
from proxima.data import BackendComparator

comparator = BackendComparator()
results = comparator.compare(
    circuit,
    backends=["quest", "cuquantum", "qsim"],
    shots=1000
)

print(results.summary())
print(f"Fastest: {results.fastest_backend}")
print(f"Result agreement: {results.fidelity:.4f}")
```

---

## Troubleshooting

### Backend Not Available

```bash
# Check why a backend is unavailable
proxima backends info cuquantum --verbose

# Common issues:
# - Missing dependencies: pip install qiskit-aer-gpu
# - No GPU: cuQuantum requires NVIDIA GPU
# - Wrong CUDA version: Check nvidia-smi
```

### Performance Issues

```bash
# Profile backend performance
proxima run --backend qsim --profile my_circuit.py

# Tips:
# - Ensure AVX2/AVX512 support for qsim
# - Check GPU memory with nvidia-smi
# - Reduce qubit count or use different backend
```

### Result Disagreement

If backends produce different results:

1. Check for numerical precision differences
2. Verify circuit is deterministic
3. Use same random seed across backends
4. Compare statevectors, not just measurement counts

---

## Next Steps

- [QuEST Installation](quest-installation.md) - Install QuEST backend
- [cuQuantum Usage](cuquantum-usage.md) - GPU acceleration guide
- [qsim Usage](qsim-usage.md) - CPU optimization guide
- [Multi-Backend Comparison](../user-guide/comparison.md) - Compare results

---

## Resources

- [Backend API Reference](../api-reference/backends/)
- [Performance Benchmarks](../developer-guide/benchmarks.md)
- [Contributing a Backend](../developer-guide/adding-backends.md)
