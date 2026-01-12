# QuEST Backend Usage Guide

> **Version:** 1.0  
> **Last Updated:** January 12, 2026  
> **Backend:** QuEST (Quantum Exact Simulation Toolkit)

---

## Overview

This guide covers how to use the QuEST backend with Proxima for high-performance quantum circuit simulations. QuEST supports both state vector and density matrix simulations with optional GPU acceleration.

---

## Quick Start

### Basic Usage via CLI

```bash
# Run a circuit with QuEST backend
proxima run --backend quest my_circuit.py

# Run with density matrix simulation
proxima run --backend quest --sim-type density_matrix my_circuit.py

# Run with GPU acceleration (if available)
proxima run --backend quest --gpu my_circuit.py
```

### Basic Usage via Python API

```python
from proxima.backends import BackendRegistry

# Get QuEST adapter
quest = BackendRegistry.get("quest")

# Check capabilities
capabilities = quest.get_capabilities()
print(f"Supports GPU: {capabilities.supports_gpu}")
print(f"Max qubits: {capabilities.max_qubits}")

# Execute a circuit
result = quest.execute(circuit, options={
    "shots": 1000,
    "sim_type": "statevector"
})
```

---

## Simulation Types

### State Vector Simulation

State vector simulation is the default mode, suitable for pure quantum states:

```python
from proxima.backends import QuestAdapter
from proxima.backends.base import SimulatorType

# Create adapter
adapter = QuestAdapter()

# Execute with state vector (default)
result = adapter.execute(
    circuit,
    options={
        "simulator_type": SimulatorType.STATE_VECTOR,
        "shots": 1024
    }
)

# Access final state vector
statevector = result.statevector
print(f"State: {statevector[:4]}...")  # First 4 amplitudes
```

**Memory Requirements:**
- Memory = 2^n × 16 bytes (for complex128)
- 20 qubits ≈ 16 MB
- 25 qubits ≈ 512 MB
- 30 qubits ≈ 16 GB

### Density Matrix Simulation

Density matrix simulation is required for mixed states and noisy simulations:

```python
from proxima.backends.base import SimulatorType

# Execute with density matrix
result = adapter.execute(
    circuit,
    options={
        "simulator_type": SimulatorType.DENSITY_MATRIX,
        "shots": 1024
    }
)

# Access density matrix
density_matrix = result.density_matrix
print(f"Trace: {density_matrix.trace():.6f}")  # Should be ~1.0
```

**Memory Requirements:**
- Memory = 2^(2n) × 16 bytes
- 10 qubits ≈ 16 MB
- 12 qubits ≈ 256 MB
- 14 qubits ≈ 4 GB

---

## Configuration Options

### Full Configuration Example

```python
options = {
    # Simulation type
    "simulator_type": SimulatorType.STATE_VECTOR,
    
    # Measurement options
    "shots": 1024,
    "seed": 42,
    
    # Precision (single, double, quad)
    "precision": "double",
    
    # GPU acceleration
    "use_gpu": True,
    "gpu_device_id": 0,
    
    # Parallelization
    "num_threads": 8,
    
    # Density matrix specific
    "truncation_threshold": 1e-4,
    
    # Result options
    "return_statevector": True,
    "return_counts": True
}

result = adapter.execute(circuit, options=options)
```

### Configuration via proxima.yaml

```yaml
backends:
  quest:
    enabled: true
    default_precision: "double"
    gpu_enabled: auto  # auto, true, false
    max_qubits: 30
    default_threads: auto  # auto or integer
    truncation_threshold: 1e-4
    
    # GPU settings
    gpu:
      device_id: 0
      memory_limit: auto
      
    # Performance tuning
    performance:
      enable_gate_fusion: true
      parallel_threshold: 15  # Use parallel mode above this qubit count
```

---

## Supported Gates

### Single-Qubit Gates

| Gate | QuEST Function | Example |
|------|----------------|---------|
| X (Pauli-X) | `pauliX` | Bit flip |
| Y (Pauli-Y) | `pauliY` | Bit+phase flip |
| Z (Pauli-Z) | `pauliZ` | Phase flip |
| H (Hadamard) | `hadamard` | Superposition |
| S | `sGate` | π/2 phase |
| T | `tGate` | π/4 phase |
| Rx(θ) | `rotateX` | X-axis rotation |
| Ry(θ) | `rotateY` | Y-axis rotation |
| Rz(θ) | `rotateZ` | Z-axis rotation |
| U3(θ,φ,λ) | `unitary` | General single-qubit |

### Two-Qubit Gates

| Gate | QuEST Function | Example |
|------|----------------|---------|
| CNOT | `controlledNot` | Controlled-X |
| CZ | `controlledPhaseFlip` | Controlled-Z |
| CRx(θ) | `controlledRotateX` | Controlled Rx |
| CRy(θ) | `controlledRotateY` | Controlled Ry |
| CRz(θ) | `controlledRotateZ` | Controlled Rz |
| SWAP | `swapGate` | Swap qubits |

### Multi-Qubit Gates

| Gate | QuEST Function | Notes |
|------|----------------|-------|
| CCX (Toffoli) | `multiControlledMultiQubitNot` | 3+ qubits |
| Multi-controlled | `multiControlledUnitary` | N controls |

---

## GPU Acceleration

### Enabling GPU

```python
# Check GPU availability
adapter = QuestAdapter()
caps = adapter.get_capabilities()

if caps.supports_gpu:
    result = adapter.execute(circuit, options={"use_gpu": True})
else:
    print("GPU not available, using CPU")
    result = adapter.execute(circuit)
```

### GPU Memory Management

```python
# Estimate GPU memory requirement
estimate = adapter.estimate_resources(circuit)
print(f"Required GPU memory: {estimate.gpu_memory_mb} MB")

# Execute with memory limit
result = adapter.execute(circuit, options={
    "use_gpu": True,
    "gpu_memory_limit_mb": 8000  # 8 GB limit
})
```

### Multi-GPU Support

```python
# Select specific GPU device
result = adapter.execute(circuit, options={
    "use_gpu": True,
    "gpu_device_id": 1  # Use second GPU
})
```

---

## Noise Simulation

QuEST supports various noise models with density matrix simulation:

### Depolarizing Noise

```python
from proxima.noise import DepolarizingNoise

# Create noisy circuit
noise_model = DepolarizingNoise(p=0.01)  # 1% error rate

result = adapter.execute(
    circuit,
    options={
        "simulator_type": SimulatorType.DENSITY_MATRIX,
        "noise_model": noise_model
    }
)
```

### Amplitude Damping

```python
from proxima.noise import AmplitudeDamping

noise_model = AmplitudeDamping(gamma=0.05)

result = adapter.execute(circuit, options={
    "simulator_type": SimulatorType.DENSITY_MATRIX,
    "noise_model": noise_model
})
```

### Custom Noise Channels

```python
from proxima.noise import KrausChannel
import numpy as np

# Define Kraus operators
K0 = np.array([[1, 0], [0, np.sqrt(1 - 0.1)]])
K1 = np.array([[0, np.sqrt(0.1)], [0, 0]])

custom_noise = KrausChannel([K0, K1])

result = adapter.execute(circuit, options={
    "simulator_type": SimulatorType.DENSITY_MATRIX,
    "noise_model": custom_noise
})
```

---

## Performance Optimization

### Thread Configuration

```python
import os

# Set before importing QuEST
os.environ['OMP_NUM_THREADS'] = '16'

# Or via options
result = adapter.execute(circuit, options={
    "num_threads": 16
})
```

### Gate Fusion

QuEST automatically fuses gates for better performance. You can control this:

```python
result = adapter.execute(circuit, options={
    "enable_gate_fusion": True,  # Default
    "fusion_max_qubits": 4  # Max qubits per fused gate
})
```

### Precision Selection

```python
# Single precision (faster, less accurate)
result = adapter.execute(circuit, options={"precision": "single"})

# Double precision (default, balanced)
result = adapter.execute(circuit, options={"precision": "double"})

# Quad precision (slowest, highest accuracy)
result = adapter.execute(circuit, options={"precision": "quad"})
```

---

## Working with Results

### Accessing Measurement Results

```python
result = adapter.execute(circuit, options={"shots": 1000})

# Get measurement counts
counts = result.counts
print(f"Counts: {counts}")
# Output: {'00': 502, '11': 498}

# Get most probable outcome
most_probable = max(counts, key=counts.get)
print(f"Most probable: {most_probable}")
```

### Accessing State Vector

```python
result = adapter.execute(circuit, options={
    "return_statevector": True
})

statevector = result.statevector
print(f"State vector shape: {statevector.shape}")
print(f"Probability of |00⟩: {abs(statevector[0])**2:.4f}")
```

### Accessing Density Matrix

```python
result = adapter.execute(circuit, options={
    "simulator_type": SimulatorType.DENSITY_MATRIX,
    "return_density_matrix": True
})

dm = result.density_matrix
purity = np.trace(dm @ dm).real
print(f"State purity: {purity:.4f}")
```

### Execution Metadata

```python
result = adapter.execute(circuit)

print(f"Backend: {result.backend_name}")
print(f"Execution time: {result.execution_time_ms:.2f} ms")
print(f"Memory used: {result.memory_mb:.2f} MB")
print(f"GPU used: {result.metadata.get('gpu_used', False)}")
print(f"Threads: {result.metadata.get('num_threads', 1)}")
```

---

## Error Handling

### Common Errors and Solutions

```python
from proxima.backends.base import (
    BackendError,
    ResourceError,
    CircuitValidationError
)

try:
    result = adapter.execute(circuit)
except ResourceError as e:
    print(f"Resource issue: {e}")
    # Try with fewer qubits or use CPU mode
except CircuitValidationError as e:
    print(f"Circuit issue: {e}")
    # Check for unsupported gates
except BackendError as e:
    print(f"Backend error: {e}")
    # General QuEST error
```

### Graceful Fallback

```python
def execute_with_fallback(circuit, options):
    """Execute with automatic fallback to CPU if GPU fails."""
    adapter = QuestAdapter()
    
    if options.get("use_gpu", False):
        try:
            return adapter.execute(circuit, options)
        except ResourceError:
            print("GPU execution failed, falling back to CPU")
            options["use_gpu"] = False
            return adapter.execute(circuit, options)
    
    return adapter.execute(circuit, options)
```

---

## Examples

### Bell State Preparation

```python
from proxima.backends import QuestAdapter
from proxima.circuits import Circuit

# Create circuit
circuit = Circuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()

# Execute
adapter = QuestAdapter()
result = adapter.execute(circuit, options={"shots": 1000})

print(f"Bell state counts: {result.counts}")
# Expected: {'00': ~500, '11': ~500}
```

### GHZ State with Density Matrix

```python
# Create GHZ circuit
circuit = Circuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)

# Execute with density matrix
result = adapter.execute(circuit, options={
    "simulator_type": SimulatorType.DENSITY_MATRIX,
    "return_density_matrix": True
})

# Verify GHZ state
dm = result.density_matrix
# GHZ state should have specific diagonal elements
print(f"ρ[0,0]: {dm[0,0]:.4f}")  # Should be ~0.5
print(f"ρ[7,7]: {dm[7,7]:.4f}")  # Should be ~0.5
```

### Variational Circuit

```python
import numpy as np

def variational_circuit(params):
    circuit = Circuit(2)
    circuit.ry(params[0], 0)
    circuit.ry(params[1], 1)
    circuit.cx(0, 1)
    circuit.ry(params[2], 0)
    circuit.ry(params[3], 1)
    return circuit

# Execute with parameters
params = np.random.random(4) * 2 * np.pi
circuit = variational_circuit(params)

result = adapter.execute(circuit, options={
    "return_statevector": True
})
```

---

## Comparison with Other Backends

| Feature | QuEST | Cirq | Qiskit Aer | qsim | cuQuantum |
|---------|-------|------|------------|------|-----------|
| State Vector | ✓ | ✓ | ✓ | ✓ | ✓ |
| Density Matrix | ✓ | ✓ | ✓ | ✗ | ✗ |
| GPU Support | ✓ | ✗ | ✓ | ✗ | ✓ |
| CPU Optimized | ✓ | ✗ | ✗ | ✓✓ | ✗ |
| Noise Support | ✓ | ✓ | ✓ | ✗ | ✗ |
| Max Qubits (typical) | 30 | 20 | 30 | 35+ | 35+ |

---

## Next Steps

- [Backend Selection Guide](backend-selection.md) - Choose the right backend
- [cuQuantum Usage](cuquantum-usage.md) - GPU-accelerated alternative
- [qsim Usage](qsim-usage.md) - CPU-optimized alternative
- [API Reference](../api-reference/backends/quest.md) - Detailed API docs

---

## Resources

- [QuEST GitHub](https://github.com/QuEST-Kit/QuEST)
- [QuEST Tutorial Paper](https://arxiv.org/abs/1802.08032)
- [Proxima Examples](../examples/quest/)
