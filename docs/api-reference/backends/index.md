# Backend API Reference

> **Version:** 1.0  
> **Last Updated:** January 12, 2026

---

## Overview

This document provides complete API reference for Proxima's backend adapter system, including the base adapter interface, individual backend adapters, and supporting data classes.

---

## Table of Contents

1. [Base Classes](#base-classes)
2. [Backend Adapters](#backend-adapters)
3. [Data Classes](#data-classes)
4. [Registry](#registry)
5. [Exceptions](#exceptions)

---

## Base Classes

### `BaseBackendAdapter`

Abstract base class that all backend adapters must implement.

**Module:** `proxima.backends.base`

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from proxima.backends.base import (
    Capabilities,
    ValidationResult,
    ResourceEstimate,
    ExecutionResult
)

class BaseBackendAdapter(ABC):
    """Abstract base class for quantum simulation backend adapters."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the unique identifier for this backend.
        
        Returns:
            str: Backend name (e.g., "quest", "cirq", "qiskit")
        """
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Return the version of the underlying backend library.
        
        Returns:
            str: Version string (e.g., "3.0.0")
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Capabilities:
        """Return the capabilities of this backend.
        
        Returns:
            Capabilities: Object describing backend capabilities
        """
        pass
    
    @abstractmethod
    def validate_circuit(self, circuit: Any) -> ValidationResult:
        """Validate if a circuit can be executed on this backend.
        
        Args:
            circuit: The quantum circuit to validate
            
        Returns:
            ValidationResult: Validation result with any issues found
        """
        pass
    
    @abstractmethod
    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        """Estimate resources required to execute the circuit.
        
        Args:
            circuit: The quantum circuit to estimate
            
        Returns:
            ResourceEstimate: Estimated memory, time, etc.
        """
        pass
    
    @abstractmethod
    def execute(
        self, 
        circuit: Any, 
        options: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute a quantum circuit on this backend.
        
        Args:
            circuit: The quantum circuit to execute
            options: Optional execution options
            
        Returns:
            ExecutionResult: Execution results including counts/statevector
        """
        pass
    
    def is_available(self) -> bool:
        """Check if this backend is available for use.
        
        Returns:
            bool: True if backend can be used
        """
        return True
    
    def cleanup(self) -> None:
        """Clean up any resources held by the adapter."""
        pass
```

---

## Backend Adapters

### `QuestAdapter`

Adapter for the QuEST (Quantum Exact Simulation Toolkit) backend.

**Module:** `proxima.backends.quest_adapter`

```python
class QuestAdapter(BaseBackendAdapter):
    """QuEST backend adapter supporting state vector and density matrix simulations."""
    
    def __init__(
        self,
        precision: str = "double",
        use_gpu: bool = False,
        num_threads: Optional[int] = None
    ):
        """Initialize the QuEST adapter.
        
        Args:
            precision: Floating point precision ("single", "double", "quad")
            use_gpu: Whether to use GPU acceleration if available
            num_threads: Number of OpenMP threads (None for auto)
        """
        pass
    
    def get_name(self) -> str:
        """Returns 'quest'."""
        return "quest"
    
    def get_version(self) -> str:
        """Returns QuEST library version."""
        pass
    
    def get_capabilities(self) -> Capabilities:
        """Returns QuEST capabilities.
        
        Returns:
            Capabilities with:
                - supports_statevector: True
                - supports_density_matrix: True
                - supports_gpu: True/False (based on build)
                - supports_noise: True
                - max_qubits: 30 (typical)
        """
        pass
    
    def validate_circuit(self, circuit: Any) -> ValidationResult:
        """Validate circuit for QuEST compatibility.
        
        Checks:
            - Gate support
            - Qubit count vs available memory
            - Parameter validity
        """
        pass
    
    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        """Estimate QuEST resource requirements.
        
        Formula:
            - State vector: 2^n * 16 bytes
            - Density matrix: 2^(2n) * 16 bytes
        """
        pass
    
    def execute(
        self,
        circuit: Any,
        options: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute circuit on QuEST.
        
        Options:
            - simulator_type: SimulatorType.STATE_VECTOR or DENSITY_MATRIX
            - shots: Number of measurement shots (default: 1024)
            - seed: Random seed for reproducibility
            - use_gpu: Override GPU setting
            - precision: Override precision setting
            - return_statevector: Include state vector in result
            - return_density_matrix: Include density matrix in result
            - truncation_threshold: For density matrix truncation
        """
        pass
    
    def cleanup(self) -> None:
        """Destroy QuEST environment and free resources."""
        pass
```

**Example Usage:**

```python
from proxima.backends import QuestAdapter
from proxima.backends.base import SimulatorType

# Create adapter
adapter = QuestAdapter(precision="double", use_gpu=True)

# Check capabilities
caps = adapter.get_capabilities()
print(f"GPU available: {caps.supports_gpu}")

# Validate circuit
validation = adapter.validate_circuit(circuit)
if not validation.is_valid:
    print(f"Issues: {validation.issues}")

# Estimate resources
estimate = adapter.estimate_resources(circuit)
print(f"Memory needed: {estimate.memory_mb} MB")

# Execute
result = adapter.execute(circuit, options={
    "simulator_type": SimulatorType.STATE_VECTOR,
    "shots": 1000,
    "return_statevector": True
})

print(f"Counts: {result.counts}")
print(f"Execution time: {result.execution_time_ms} ms")

# Cleanup
adapter.cleanup()
```

---

### `CuQuantumAdapter`

Adapter for NVIDIA cuQuantum GPU-accelerated simulations.

**Module:** `proxima.backends.cuquantum_adapter`

```python
class CuQuantumAdapter(BaseBackendAdapter):
    """cuQuantum backend adapter for GPU-accelerated state vector simulations."""
    
    def __init__(
        self,
        device_id: int = 0,
        memory_limit: Optional[int] = None
    ):
        """Initialize the cuQuantum adapter.
        
        Args:
            device_id: CUDA device ID to use
            memory_limit: GPU memory limit in MB (None for auto)
            
        Raises:
            BackendNotAvailableError: If no NVIDIA GPU is found
        """
        pass
    
    def get_name(self) -> str:
        """Returns 'cuquantum'."""
        return "cuquantum"
    
    def get_version(self) -> str:
        """Returns cuQuantum SDK version."""
        pass
    
    def get_capabilities(self) -> Capabilities:
        """Returns cuQuantum capabilities.
        
        Returns:
            Capabilities with:
                - supports_statevector: True
                - supports_density_matrix: False
                - supports_gpu: True
                - supports_noise: False
                - max_qubits: 35+ (depends on GPU memory)
        """
        pass
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get information about the GPU being used.
        
        Returns:
            Dict with:
                - name: GPU model name
                - compute_capability: CUDA compute capability
                - total_memory_mb: Total GPU memory
                - free_memory_mb: Available GPU memory
        """
        pass
    
    def validate_circuit(self, circuit: Any) -> ValidationResult:
        """Validate circuit for cuQuantum compatibility.
        
        Checks:
            - GPU memory vs state vector size
            - Gate support
            - No noise model (not supported)
        """
        pass
    
    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        """Estimate cuQuantum resource requirements.
        
        Formula:
            GPU Memory = 2^n * 16 + workspace (1-2 GB) + overhead (500 MB)
        """
        pass
    
    def execute(
        self,
        circuit: Any,
        options: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute circuit on cuQuantum.
        
        Options:
            - shots: Number of measurement shots
            - seed: Random seed
            - return_statevector: Include state vector
            - blocking: Wait for GPU completion (default: True)
        """
        pass
```

**Example Usage:**

```python
from proxima.backends import CuQuantumAdapter

# Create adapter (will fail if no GPU)
adapter = CuQuantumAdapter(device_id=0)

# Check GPU info
gpu_info = adapter.get_gpu_info()
print(f"Using: {gpu_info['name']}")
print(f"Free memory: {gpu_info['free_memory_mb']} MB")

# Execute large circuit
result = adapter.execute(large_circuit, options={
    "shots": 10000,
    "return_statevector": True
})

print(f"GPU execution time: {result.execution_time_ms} ms")
```

---

### `QsimAdapter`

Adapter for Google's qsim CPU-optimized simulator.

**Module:** `proxima.backends.qsim_adapter`

```python
class QsimAdapter(BaseBackendAdapter):
    """qsim backend adapter for CPU-optimized state vector simulations."""
    
    def __init__(
        self,
        num_threads: Optional[int] = None,
        enable_gate_fusion: bool = True
    ):
        """Initialize the qsim adapter.
        
        Args:
            num_threads: Number of threads (None for auto based on CPU)
            enable_gate_fusion: Enable automatic gate fusion optimization
        """
        pass
    
    def get_name(self) -> str:
        """Returns 'qsim'."""
        return "qsim"
    
    def get_version(self) -> str:
        """Returns qsim version."""
        pass
    
    def get_capabilities(self) -> Capabilities:
        """Returns qsim capabilities.
        
        Returns:
            Capabilities with:
                - supports_statevector: True
                - supports_density_matrix: False
                - supports_gpu: False
                - supports_noise: False
                - supports_avx2: True/False
                - supports_avx512: True/False
                - max_qubits: 35+
        """
        pass
    
    def validate_circuit(self, circuit: Any) -> ValidationResult:
        """Validate circuit for qsim compatibility.
        
        Checks:
            - Gate support (some gates need decomposition)
            - No mid-circuit measurements
            - No noise model
        """
        pass
    
    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        """Estimate qsim resource requirements.
        
        Formula:
            Memory = 2^n * 16 bytes (state vector)
            Time ≈ depth * gate_count * factor
        """
        pass
    
    def execute(
        self,
        circuit: Any,
        options: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute circuit on qsim.
        
        Options:
            - shots: Number of measurement shots
            - seed: Random seed
            - num_threads: Override thread count
            - enable_gate_fusion: Override fusion setting
            - return_statevector: Include state vector
        """
        pass
```

---

## Data Classes

### `Capabilities`

Describes what a backend can do.

**Module:** `proxima.backends.base`

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Capabilities:
    """Backend capability description."""
    
    # Simulation types supported
    supports_statevector: bool = True
    supports_density_matrix: bool = False
    
    # Hardware acceleration
    supports_gpu: bool = False
    supports_multi_gpu: bool = False
    
    # Features
    supports_noise: bool = False
    supports_mid_circuit_measurement: bool = False
    supports_classical_control: bool = False
    
    # Limits
    max_qubits: int = 20
    max_shots: int = 1_000_000
    
    # Optimization
    supports_gate_fusion: bool = False
    supports_avx2: bool = False
    supports_avx512: bool = False
    
    # Precision
    supported_precisions: List[str] = None  # ["single", "double", "quad"]
    
    # Additional info
    version: str = ""
    additional_info: Optional[dict] = None
    
    def __post_init__(self):
        if self.supported_precisions is None:
            self.supported_precisions = ["double"]
```

---

### `ValidationResult`

Result of circuit validation.

```python
from dataclasses import dataclass, field
from typing import List
from enum import Enum

class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class ValidationIssue:
    """A single validation issue."""
    message: str
    severity: ValidationSeverity
    location: Optional[str] = None  # e.g., "gate 5", "qubit 3"
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of circuit validation."""
    
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(i.severity == ValidationSeverity.ERROR for i in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)
    
    def __str__(self) -> str:
        if self.is_valid:
            return "Valid"
        return f"Invalid: {len(self.issues)} issue(s)"
```

---

### `ResourceEstimate`

Estimated resources for circuit execution.

```python
@dataclass
class ResourceEstimate:
    """Estimated resources required for circuit execution."""
    
    # Memory requirements (in MB)
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    
    # Time estimates (in seconds)
    estimated_time_seconds: Optional[float] = None
    
    # Complexity metrics
    num_qubits: int = 0
    circuit_depth: int = 0
    gate_count: int = 0
    
    # Feasibility
    is_feasible: bool = True
    feasibility_reason: Optional[str] = None
    
    # Recommendations
    recommended_backend: Optional[str] = None
    recommendation_reason: Optional[str] = None
```

---

### `ExecutionResult`

Result of circuit execution.

```python
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np

@dataclass
class ExecutionResult:
    """Result of executing a quantum circuit."""
    
    # Core results
    counts: Dict[str, int] = field(default_factory=dict)
    statevector: Optional[np.ndarray] = None
    density_matrix: Optional[np.ndarray] = None
    
    # Execution info
    backend_name: str = ""
    execution_time_ms: float = 0.0
    
    # Resource usage
    memory_mb: float = 0.0
    gpu_memory_mb: Optional[float] = None
    
    # Options used
    shots: int = 0
    simulator_type: str = "statevector"
    
    # Status
    success: bool = True
    error_message: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def probabilities(self) -> Dict[str, float]:
        """Convert counts to probabilities."""
        total = sum(self.counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.counts.items()}
    
    def get_expectation_value(self, observable: np.ndarray) -> complex:
        """Calculate expectation value of an observable.
        
        Args:
            observable: Hermitian operator as numpy array
            
        Returns:
            Expectation value <ψ|O|ψ>
        """
        if self.statevector is not None:
            return np.vdot(self.statevector, observable @ self.statevector)
        elif self.density_matrix is not None:
            return np.trace(observable @ self.density_matrix)
        else:
            raise ValueError("No state available for expectation value")
```

---

### `SimulatorType`

Enum for simulation types.

```python
from enum import Enum

class SimulatorType(Enum):
    """Type of quantum simulation."""
    STATE_VECTOR = "statevector"
    DENSITY_MATRIX = "density_matrix"

class ResultType(Enum):
    """Type of result to return."""
    COUNTS = "counts"
    STATEVECTOR = "statevector"
    DENSITY_MATRIX = "density_matrix"
    EXPECTATION_VALUE = "expectation_value"
```

---

## Registry

### `BackendRegistry`

Central registry for backend discovery and access.

**Module:** `proxima.backends.registry`

```python
class BackendRegistry:
    """Registry for discovering and accessing backend adapters."""
    
    @classmethod
    def discover(cls) -> Dict[str, BaseBackendAdapter]:
        """Discover all available backends.
        
        Returns:
            Dict mapping backend names to adapter instances
        """
        pass
    
    @classmethod
    def get(cls, name: str) -> BaseBackendAdapter:
        """Get a specific backend adapter.
        
        Args:
            name: Backend name (e.g., "quest", "cirq")
            
        Returns:
            Backend adapter instance
            
        Raises:
            BackendNotFoundError: If backend not found
            BackendNotAvailableError: If backend not available
        """
        pass
    
    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if a backend is available.
        
        Args:
            name: Backend name
            
        Returns:
            True if backend can be used
        """
        pass
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List names of all available backends.
        
        Returns:
            List of backend names
        """
        pass
    
    @classmethod
    def register(
        cls, 
        name: str, 
        adapter_class: type,
        priority: int = 50
    ) -> None:
        """Register a custom backend adapter.
        
        Args:
            name: Unique backend name
            adapter_class: Adapter class (subclass of BaseBackendAdapter)
            priority: Selection priority (higher = preferred)
        """
        pass
```

---

## Exceptions

### Backend Exceptions

```python
class BackendError(Exception):
    """Base exception for backend errors."""
    pass

class BackendNotFoundError(BackendError):
    """Backend with given name not found in registry."""
    pass

class BackendNotAvailableError(BackendError):
    """Backend exists but is not available (missing dependencies, etc.)."""
    
    def __init__(self, backend_name: str, reason: str):
        self.backend_name = backend_name
        self.reason = reason
        super().__init__(f"Backend '{backend_name}' not available: {reason}")

class CircuitValidationError(BackendError):
    """Circuit failed validation for a backend."""
    
    def __init__(self, validation_result: ValidationResult):
        self.validation_result = validation_result
        issues = "; ".join(str(i.message) for i in validation_result.issues)
        super().__init__(f"Circuit validation failed: {issues}")

class ResourceError(BackendError):
    """Insufficient resources for execution."""
    
    def __init__(self, required: ResourceEstimate, available: Dict[str, Any]):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient resources: need {required.memory_mb}MB, "
            f"have {available.get('memory_mb', 'unknown')}MB"
        )

class ExecutionError(BackendError):
    """Error during circuit execution."""
    
    def __init__(self, backend_name: str, message: str, original_error: Exception = None):
        self.backend_name = backend_name
        self.original_error = original_error
        super().__init__(f"Execution failed on {backend_name}: {message}")
```

---

## Configuration Schema

### Backend Configuration

```yaml
# proxima.yaml - Backend configuration section

backends:
  # Global settings
  default: auto  # Backend to use when not specified
  auto_selection:
    enabled: true
    explain_selection: true
    performance_history: true
  
  # Fallback settings
  fallback:
    enabled: true
    max_attempts: 3
  
  # Individual backend configurations
  quest:
    enabled: true
    default_precision: "double"  # single, double, quad
    gpu_enabled: auto  # auto, true, false
    max_qubits: 30
    num_threads: auto  # auto or integer
    truncation_threshold: 1.0e-4
    
  cuquantum:
    enabled: true
    device_id: 0
    memory_limit: auto  # auto or MB value
    workspace_size_mb: 1024
    
  qsim:
    enabled: true
    num_threads: auto
    enable_gate_fusion: true
    max_fused_qubits: 4
    
  cirq:
    enabled: true
    max_qubits: 20
    
  qiskit:
    enabled: true
    optimization_level: 2
    max_qubits: 30
    
  lret:
    enabled: true
    max_rank: 64
    convergence_threshold: 1.0e-6
```

---

## Result Format Examples

### State Vector Result

```python
{
    "counts": {"00": 512, "11": 488},
    "statevector": [0.707+0j, 0+0j, 0+0j, 0.707+0j],
    "backend_name": "quest",
    "execution_time_ms": 15.3,
    "memory_mb": 0.016,
    "shots": 1000,
    "simulator_type": "statevector",
    "success": True,
    "metadata": {
        "precision": "double",
        "num_threads": 8,
        "gpu_used": False,
        "gate_fusion_applied": True
    }
}
```

### Density Matrix Result

```python
{
    "counts": {"00": 489, "01": 11, "10": 12, "11": 488},
    "density_matrix": [[0.5, 0, 0, 0.49], ...],
    "backend_name": "quest",
    "execution_time_ms": 45.7,
    "memory_mb": 0.256,
    "shots": 1000,
    "simulator_type": "density_matrix",
    "success": True,
    "metadata": {
        "final_rank": 4,
        "truncation_applied": False,
        "noise_model": "depolarizing_0.01"
    }
}
```

### GPU Execution Result

```python
{
    "counts": {"0" * 30: 1000},
    "backend_name": "cuquantum",
    "execution_time_ms": 234.5,
    "gpu_memory_mb": 17408.0,
    "shots": 1000,
    "simulator_type": "statevector",
    "success": True,
    "metadata": {
        "gpu_name": "NVIDIA A100",
        "gpu_utilization": 0.85,
        "cuda_version": "12.0"
    }
}
```

---

## See Also

- [Backend Selection Guide](../backends/backend-selection.md)
- [QuEST Installation](../backends/quest-installation.md)
- [QuEST Usage](../backends/quest-usage.md)
- [Developer Guide: Adding Backends](../developer-guide/adding-backends.md)
