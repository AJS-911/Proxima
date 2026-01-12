# Migration Guide: Adding New Backends

> **Version:** 1.0  
> **Last Updated:** January 12, 2026  
> **Purpose:** Guide for upgrading Proxima and understanding new backend features

---

## Overview

This guide helps existing Proxima users upgrade to versions with new backend support (QuEST, cuQuantum, qsim) and provides information for developers adding custom backends.

---

## Table of Contents

1. [For Existing Users](#for-existing-users)
2. [What Changed](#what-changed)
3. [Upgrade Instructions](#upgrade-instructions)
4. [Troubleshooting](#troubleshooting)
5. [For Developers: Adding Custom Backends](#for-developers-adding-custom-backends)

---

## For Existing Users

### What's New

Proxima now supports three additional high-performance backends:

| Backend | Description | Best For |
|---------|-------------|----------|
| **QuEST** | High-performance C++ simulator | Research, GPU/CPU flexibility, density matrix |
| **cuQuantum** | NVIDIA GPU-accelerated | Large circuits on NVIDIA GPUs |
| **qsim** | Google's CPU-optimized simulator | Large circuits on CPU, Cirq users |

### Do I Need to Change Anything?

**Short answer:** No, your existing code and workflows will continue to work.

**Longer answer:** 
- If you use `--backend auto` (default), Proxima may now select a new backend if it's better suited for your circuit
- Your explicit backend selections (`--backend cirq`, `--backend qiskit`) will continue to work
- No breaking changes to the API or CLI

### Benefits of Upgrading

- **Performance:** Up to 10x faster execution for large circuits
- **Scale:** Support for 35+ qubits (up from ~20-25)
- **GPU:** Better GPU acceleration via cuQuantum
- **Flexibility:** More backend choices for different use cases

---

## What Changed

### New Files Added

```
src/proxima/backends/
├── quest_adapter.py       ← NEW: QuEST backend adapter
├── cuquantum_adapter.py   ← NEW: cuQuantum backend adapter
└── qsim_adapter.py        ← NEW: qsim backend adapter
```

### Modified Files

| File | Changes |
|------|---------|
| `backends/registry.py` | Added discovery for new backends |
| `backends/base.py` | Extended Capabilities dataclass (no breaking changes) |
| `intelligence/selector.py` | Updated auto-selection algorithm |
| `config/defaults.py` | Added configuration for new backends |

### New Dependencies (Optional)

```txt
# Only install what you need:
pyquest-cffi>=0.9.0       # For QuEST backend
qiskit-aer-gpu>=0.13.0    # For cuQuantum backend  
qsimcirq>=0.22.0          # For qsim backend
```

### Configuration Changes

New configuration options are available but all have sensible defaults:

```yaml
# New options in proxima.yaml (all optional)
backends:
  quest:
    enabled: true
    default_precision: "double"
    gpu_enabled: auto
    
  cuquantum:
    enabled: true
    device_id: 0
    
  qsim:
    enabled: true
    num_threads: auto
    enable_gate_fusion: true
```

### Auto-Selection Algorithm Changes

The backend auto-selection now considers:

1. **GPU availability:** If NVIDIA GPU is present, cuQuantum is preferred for large state vector simulations
2. **Circuit size:** For 25+ qubits, qsim or cuQuantum are preferred
3. **Simulation type:** QuEST is preferred for density matrix simulations
4. **CPU features:** qsim is preferred if AVX2/AVX512 is available

**Previous behavior:** Cirq or Qiskit based on circuit format  
**New behavior:** Best-performance backend based on circuit and hardware analysis

To restore previous behavior:
```yaml
backends:
  auto_selection:
    legacy_mode: true  # Use old selection algorithm
```

---

## Upgrade Instructions

### Step 1: Update Proxima

```bash
# If installed via pip
pip install --upgrade proxima

# If installed from source
cd proxima
git pull origin main
pip install -e .
```

### Step 2: Install New Backend Dependencies (Optional)

Only install the backends you want to use:

```bash
# For QuEST (high-performance, GPU/CPU)
pip install pyquest-cffi>=0.9.0

# For cuQuantum (NVIDIA GPU)
pip install qiskit-aer-gpu>=0.13.0

# For qsim (CPU-optimized)
pip install qsimcirq>=0.22.0
```

### Step 3: Verify Installation

```bash
# Check available backends
proxima backends list

# Test new backends
proxima backends test --all

# Get detailed info
proxima backends info quest
proxima backends info cuquantum
proxima backends info qsim
```

### Step 4: Update Configuration (Optional)

If you want to customize new backend settings:

```bash
# Generate new config with all options
proxima config init --force

# Or manually add to existing config
proxima config set backends.quest.gpu_enabled true
proxima config set backends.qsim.num_threads 16
```

### Step 5: Test Your Workflows

```bash
# Run your existing circuits
proxima run my_circuit.py

# Check which backend was selected
proxima run --verbose my_circuit.py

# Compare old vs new backends
proxima compare --backends cirq,qsim my_circuit.py
```

---

## Troubleshooting

### Common Issues After Upgrade

#### 1. Backend Not Detected

**Symptom:** `proxima backends list` doesn't show new backend

**Cause:** Missing dependencies

**Solution:**
```bash
# Check why backend is unavailable
proxima backends info quest --verbose

# Install missing dependency
pip install pyquest-cffi
```

#### 2. GPU Not Detected for cuQuantum

**Symptom:** cuQuantum shows "GPU not available"

**Cause:** CUDA not properly installed or no NVIDIA GPU

**Solution:**
```bash
# Verify NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Ensure CUDA paths are set
echo $CUDA_HOME  # Should point to CUDA installation
```

#### 3. Different Results After Upgrade

**Symptom:** Results differ slightly from before

**Cause:** Different random number generation or numerical precision

**Solution:**
```bash
# Use explicit seed for reproducibility
proxima run --seed 42 my_circuit.py

# Use specific backend for consistency
proxima run --backend cirq my_circuit.py
```

#### 4. Performance Worse Than Expected

**Symptom:** New backend is slower than expected

**Causes and Solutions:**

1. **Small circuits:** New backends have more initialization overhead
   - Solution: Use original backends for < 10 qubits

2. **AVX not utilized:** qsim not using vectorization
   - Check: `proxima backends info qsim` (should show AVX2/AVX512)
   - Solution: Ensure CPU supports AVX2

3. **GPU memory insufficient:** cuQuantum falling back to CPU
   - Check: `nvidia-smi` during execution
   - Solution: Reduce qubit count or use qsim instead

#### 5. QuEST Build Failures

**Symptom:** pyquest-cffi installation fails

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake libeigen3-dev

# macOS
brew install cmake eigen

# Windows: Install Visual Studio Build Tools, then:
pip install pyquest-cffi
```

---

## For Developers: Adding Custom Backends

### Creating a New Backend Adapter

Follow these steps to add your own backend:

#### Step 1: Create Adapter Class

Create a new file in `src/proxima/backends/`:

```python
# src/proxima/backends/my_backend_adapter.py

from typing import Optional, Dict, Any
from proxima.backends.base import (
    BaseBackendAdapter,
    Capabilities,
    ValidationResult,
    ResourceEstimate,
    ExecutionResult,
    SimulatorType
)

class MyBackendAdapter(BaseBackendAdapter):
    """Adapter for MyBackend quantum simulator."""
    
    def __init__(self, **kwargs):
        """Initialize the adapter."""
        self._my_backend = None
        self._initialize(**kwargs)
    
    def _initialize(self, **kwargs):
        """Initialize the underlying backend."""
        try:
            import my_backend_library
            self._my_backend = my_backend_library.Simulator()
        except ImportError:
            raise ImportError(
                "my_backend_library not installed. "
                "Install with: pip install my-backend-library"
            )
    
    def get_name(self) -> str:
        return "mybackend"
    
    def get_version(self) -> str:
        import my_backend_library
        return my_backend_library.__version__
    
    def get_capabilities(self) -> Capabilities:
        return Capabilities(
            supports_statevector=True,
            supports_density_matrix=False,
            supports_gpu=False,
            supports_noise=False,
            max_qubits=25,
            supported_precisions=["double"]
        )
    
    def is_available(self) -> bool:
        try:
            import my_backend_library
            return True
        except ImportError:
            return False
    
    def validate_circuit(self, circuit: Any) -> ValidationResult:
        issues = []
        # Add validation logic
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues
        )
    
    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        num_qubits = circuit.num_qubits
        memory_mb = (2 ** num_qubits) * 16 / (1024 * 1024)
        return ResourceEstimate(
            memory_mb=memory_mb,
            num_qubits=num_qubits,
            circuit_depth=circuit.depth,
            is_feasible=memory_mb < 16000  # 16 GB limit
        )
    
    def execute(
        self,
        circuit: Any,
        options: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        options = options or {}
        shots = options.get("shots", 1024)
        
        import time
        start = time.time()
        
        # Convert circuit to backend format
        backend_circuit = self._convert_circuit(circuit)
        
        # Execute
        result = self._my_backend.run(backend_circuit, shots=shots)
        
        execution_time = (time.time() - start) * 1000
        
        return ExecutionResult(
            counts=result.get_counts(),
            backend_name=self.get_name(),
            execution_time_ms=execution_time,
            shots=shots,
            success=True
        )
    
    def _convert_circuit(self, circuit):
        """Convert Proxima circuit to backend format."""
        # Implementation depends on your backend
        pass
    
    def cleanup(self):
        """Clean up resources."""
        if self._my_backend:
            self._my_backend.close()
```

#### Step 2: Register in Registry

Add to `src/proxima/backends/registry.py`:

```python
# In the discover() method, add:

def discover(cls) -> Dict[str, BaseBackendAdapter]:
    backends = {}
    
    # ... existing backends ...
    
    # Try to load MyBackend
    try:
        from proxima.backends.my_backend_adapter import MyBackendAdapter
        adapter = MyBackendAdapter()
        if adapter.is_available():
            backends["mybackend"] = adapter
    except ImportError as e:
        logger.debug(f"MyBackend not available: {e}")
    
    return backends
```

#### Step 3: Add Configuration

Add to `src/proxima/config/defaults.py`:

```python
# In BACKEND_DEFAULTS dict:
"mybackend": {
    "enabled": True,
    "max_qubits": 25,
    # ... your options ...
}
```

#### Step 4: Write Tests

Create `tests/backends/test_my_backend_adapter.py`:

```python
import pytest
from unittest.mock import MagicMock, patch

class TestMyBackendAdapter:
    """Tests for MyBackend adapter."""
    
    @pytest.fixture
    def mock_my_backend(self):
        with patch("my_backend_library") as mock:
            yield mock
    
    def test_get_name(self, mock_my_backend):
        from proxima.backends.my_backend_adapter import MyBackendAdapter
        adapter = MyBackendAdapter()
        assert adapter.get_name() == "mybackend"
    
    def test_execute_bell_state(self, mock_my_backend):
        # Test execution
        pass
    
    # ... more tests ...
```

#### Step 5: Document Your Backend

Create documentation in `docs/backends/mybackend-installation.md` and `docs/backends/mybackend-usage.md`.

### Backend Development Best Practices

1. **Graceful Degradation:** Always check if dependencies are available
2. **Error Messages:** Provide clear error messages with installation instructions
3. **Resource Cleanup:** Implement `cleanup()` to prevent memory leaks
4. **Validation:** Thoroughly validate circuits before execution
5. **Logging:** Use structured logging for debugging
6. **Testing:** Write comprehensive unit and integration tests
7. **Documentation:** Document capabilities, limitations, and usage

---

## Breaking Changes

### Version 1.0.0 (This Release)

**No breaking changes.** All existing code and configurations continue to work.

### Future Deprecations

The following are planned for deprecation in version 2.0.0:

| Deprecated | Replacement | Timeline |
|------------|-------------|----------|
| `--simulator` flag | `--sim-type` flag | v2.0.0 |
| `backend.simulate()` | `backend.execute()` | v2.0.0 |

---

## Getting Help

- **Documentation:** [docs.proxima.dev](https://docs.proxima.dev)
- **GitHub Issues:** [github.com/proxima/proxima/issues](https://github.com/proxima/proxima/issues)
- **Discussions:** [github.com/proxima/proxima/discussions](https://github.com/proxima/proxima/discussions)

---

## See Also

- [Backend Selection Guide](../backends/backend-selection.md)
- [QuEST Installation](../backends/quest-installation.md)
- [API Reference](../api-reference/backends/)
