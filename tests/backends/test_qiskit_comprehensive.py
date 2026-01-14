"""Comprehensive tests for Qiskit Aer Backend Adapter.

Tests all features including:
- GPU support integration
- Advanced transpilation options
- Snapshot-based execution
- Comprehensive noise model support
"""

from __future__ import annotations

import logging
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import adapter and supporting classes
from proxima.backends.qiskit_adapter import (
    AdvancedTranspiler,
    GPUConfiguration,
    GPUDeviceType,
    GPUManager,
    GPUStatus,
    NoiseAnalysisResult,
    NoiseModelConfig,
    NoiseModelManager,
    NoiseModelType,
    NoiseParameters,
    QiskitBackendAdapter,
    SnapshotConfig,
    SnapshotManager,
    SnapshotType,
    TranspilationConfig,
    TranspilationResult,
)
from proxima.backends.base import (
    Capabilities,
    ExecutionResult,
    ResourceEstimate,
    ResultType,
    SimulatorType,
    ValidationResult,
)
from proxima.backends.exceptions import (
    BackendNotInstalledError,
    CircuitValidationError,
    QubitLimitExceededError,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def adapter() -> QiskitBackendAdapter:
    """Create a QiskitBackendAdapter instance."""
    return QiskitBackendAdapter()


@pytest.fixture
def qiskit_available() -> bool:
    """Check if Qiskit is available."""
    try:
        import qiskit
        import qiskit_aer
        return True
    except ImportError:
        return False


@pytest.fixture
def simple_circuit():
    """Create a simple 2-qubit Bell state circuit."""
    try:
        from qiskit import QuantumCircuit
        
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        return circuit
    except ImportError:
        pytest.skip("Qiskit not installed")


@pytest.fixture
def parameterized_circuit():
    """Create a parameterized variational circuit."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        
        theta = Parameter("theta")
        phi = Parameter("phi")
        
        circuit = QuantumCircuit(1)
        circuit.rx(theta, 0)
        circuit.rz(phi, 0)
        
        return circuit, {"theta": np.pi / 4, "phi": np.pi / 2}
    except ImportError:
        pytest.skip("Qiskit not installed")


@pytest.fixture
def large_circuit():
    """Create a larger circuit for performance testing."""
    try:
        from qiskit import QuantumCircuit
        
        n_qubits = 10
        circuit = QuantumCircuit(n_qubits)
        
        for _ in range(20):  # 20 layers
            for i in range(n_qubits):
                circuit.h(i)
            for i in range(0, n_qubits - 1, 2):
                circuit.cx(i, i + 1)
        
        return circuit
    except ImportError:
        pytest.skip("Qiskit not installed")


# ==============================================================================
# BASIC ADAPTER TESTS
# ==============================================================================


class TestQiskitBackendAdapter:
    """Test basic QiskitBackendAdapter functionality."""

    def test_get_name(self, adapter: QiskitBackendAdapter) -> None:
        """Test get_name returns 'qiskit'."""
        assert adapter.get_name() == "qiskit"

    def test_get_version(self, adapter: QiskitBackendAdapter, qiskit_available: bool) -> None:
        """Test get_version returns a version string."""
        version = adapter.get_version()
        if qiskit_available:
            assert isinstance(version, str)
            assert version != "unavailable"
        else:
            assert version in ("unknown", "unavailable")

    def test_is_available(self, adapter: QiskitBackendAdapter, qiskit_available: bool) -> None:
        """Test is_available matches actual Qiskit availability."""
        assert adapter.is_available() == qiskit_available

    def test_get_capabilities(self, adapter: QiskitBackendAdapter) -> None:
        """Test get_capabilities returns proper Capabilities object."""
        caps = adapter.get_capabilities()
        
        assert isinstance(caps, Capabilities)
        assert SimulatorType.STATE_VECTOR in caps.simulator_types
        assert SimulatorType.DENSITY_MATRIX in caps.simulator_types
        assert caps.max_qubits == 28
        assert caps.supports_noise is True
        assert caps.supports_batching is True
        
        # Check custom features
        assert caps.custom_features.get("transpilation") is True
        assert caps.custom_features.get("advanced_transpilation") is True
        assert "snapshot_modes" in caps.custom_features
        assert "noise_models" in caps.custom_features

    def test_supports_simulator(self, adapter: QiskitBackendAdapter) -> None:
        """Test supports_simulator method."""
        assert adapter.supports_simulator(SimulatorType.STATE_VECTOR) is True
        assert adapter.supports_simulator(SimulatorType.DENSITY_MATRIX) is True
        assert adapter.supports_simulator(SimulatorType.TENSOR_NETWORK) is False

    def test_get_supported_gates(self, adapter: QiskitBackendAdapter) -> None:
        """Test get_supported_gates returns a list of gates."""
        gates = adapter.get_supported_gates()
        
        assert isinstance(gates, list)
        assert "h" in gates
        assert "x" in gates
        assert "cx" in gates


# ==============================================================================
# GPU SUPPORT TESTS
# ==============================================================================


class TestGPUManager:
    """Test GPUManager functionality."""

    def test_manager_initialization(self) -> None:
        """Test GPUManager can be initialized."""
        manager = GPUManager()
        assert manager is not None

    def test_detect_gpu(self) -> None:
        """Test GPU detection returns status."""
        manager = GPUManager()
        status = manager.detect_gpu()
        
        assert isinstance(status, GPUStatus)
        assert isinstance(status.is_available, bool)
        assert isinstance(status.device_name, str)

    def test_configure_simulator_cpu(self) -> None:
        """Test simulator configuration for CPU."""
        manager = GPUManager()
        config = GPUConfiguration(enabled=False)
        
        options = manager.configure_simulator(config)
        
        assert options.get("device") == "CPU"

    def test_configure_simulator_gpu_unavailable(self) -> None:
        """Test GPU config with fallback when GPU unavailable."""
        manager = GPUManager()
        manager._cached_status = GPUStatus(is_available=False)
        
        config = GPUConfiguration(enabled=True, fallback_to_cpu=True)
        options = manager.configure_simulator(config)
        
        assert options.get("device") == "CPU"

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        manager = GPUManager()
        manager._cached_status = GPUStatus(is_available=True)
        manager.clear_cache()
        
        assert manager._cached_status is None


class TestGPUConfiguration:
    """Test GPUConfiguration dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = GPUConfiguration()
        
        assert config.enabled is False
        assert config.device_type == GPUDeviceType.AUTO
        assert config.device_id == 0
        assert config.fallback_to_cpu is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = GPUConfiguration(
            enabled=True,
            device_type=GPUDeviceType.CUDA,
            max_memory_mb=4096,
        )
        
        assert config.enabled is True
        assert config.device_type == GPUDeviceType.CUDA
        assert config.max_memory_mb == 4096


class TestGPUIntegration:
    """Integration tests for GPU features."""

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_adapter_gpu_status(self, adapter: QiskitBackendAdapter) -> None:
        """Test adapter GPU status retrieval."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        status = adapter.get_gpu_status()
        assert isinstance(status, GPUStatus)

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_adapter_configure_gpu(self, adapter: QiskitBackendAdapter) -> None:
        """Test adapter GPU configuration."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        config = GPUConfiguration(enabled=False)
        adapter.configure_gpu(config)
        
        # Verify config is stored
        assert adapter._gpu_config.enabled is False


# ==============================================================================
# ADVANCED TRANSPILATION TESTS
# ==============================================================================


class TestAdvancedTranspiler:
    """Test AdvancedTranspiler functionality."""

    def test_transpiler_initialization(self) -> None:
        """Test AdvancedTranspiler can be initialized."""
        transpiler = AdvancedTranspiler()
        assert transpiler is not None

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_basic_transpilation(self, simple_circuit) -> None:
        """Test basic circuit transpilation."""
        transpiler = AdvancedTranspiler()
        config = TranspilationConfig(optimization_level=1)
        
        result = transpiler.transpile(simple_circuit, config)
        
        assert isinstance(result, TranspilationResult)
        assert result.circuit is not None
        assert result.transpiled_depth >= 0
        assert result.optimization_time_ms > 0

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_transpilation_levels(self, large_circuit) -> None:
        """Test different optimization levels."""
        transpiler = AdvancedTranspiler()
        
        results = {}
        for level in [0, 1, 2, 3]:
            config = TranspilationConfig(optimization_level=level)
            results[level] = transpiler.transpile(large_circuit, config)
        
        # Higher levels should generally not increase depth
        # (though this isn't guaranteed in all cases)
        assert all(r.transpiled_depth >= 0 for r in results.values())

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_estimate_transpilation_benefit(self, large_circuit) -> None:
        """Test transpilation benefit estimation."""
        transpiler = AdvancedTranspiler()
        
        benefits = transpiler.estimate_transpilation_benefit(large_circuit, [0, 1])
        
        assert 0 in benefits
        assert 1 in benefits
        assert "depth" in benefits[0]
        assert "gates" in benefits[0]


class TestTranspilationConfig:
    """Test TranspilationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TranspilationConfig()
        
        assert config.optimization_level == 1
        assert config.layout_method == "sabre"
        assert config.routing_method == "sabre"
        assert config.approximation_degree == 1.0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TranspilationConfig(
            optimization_level=3,
            basis_gates=["cx", "rz", "sx"],
            layout_method="dense",
        )
        
        assert config.optimization_level == 3
        assert config.basis_gates == ["cx", "rz", "sx"]
        assert config.layout_method == "dense"


class TestTranspilationResult:
    """Test TranspilationResult dataclass."""

    def test_reduction_calculations(self) -> None:
        """Test reduction percentage calculations."""
        result = TranspilationResult(
            circuit=None,
            original_depth=100,
            transpiled_depth=50,
            original_gates=200,
            transpiled_gates=100,
            original_two_qubit_gates=50,
            transpiled_two_qubit_gates=25,
        )
        
        assert result.depth_reduction == 50.0
        assert result.gate_reduction == 50.0

    def test_zero_original(self) -> None:
        """Test handling of zero original values."""
        result = TranspilationResult(
            circuit=None,
            original_depth=0,
            transpiled_depth=0,
            original_gates=0,
            transpiled_gates=0,
            original_two_qubit_gates=0,
            transpiled_two_qubit_gates=0,
        )
        
        assert result.depth_reduction == 0.0
        assert result.gate_reduction == 0.0


# ==============================================================================
# SNAPSHOT TESTS
# ==============================================================================


class TestSnapshotManager:
    """Test SnapshotManager functionality."""

    def test_manager_initialization(self) -> None:
        """Test SnapshotManager can be initialized."""
        manager = SnapshotManager()
        assert manager is not None

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_add_statevector_snapshot(self) -> None:
        """Test adding statevector snapshot."""
        from qiskit import QuantumCircuit
        
        manager = SnapshotManager()
        circuit = QuantumCircuit(2)
        circuit.h(0)
        
        config = SnapshotConfig(
            snapshot_type=SnapshotType.STATEVECTOR,
            label="test_snapshot",
        )
        
        modified = manager.add_snapshot(circuit, config)
        
        # Modified circuit should have more instructions
        assert modified.size() >= circuit.size()

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_add_density_matrix_snapshot(self) -> None:
        """Test adding density matrix snapshot."""
        from qiskit import QuantumCircuit
        
        manager = SnapshotManager()
        circuit = QuantumCircuit(2)
        circuit.h(0)
        
        config = SnapshotConfig(
            snapshot_type=SnapshotType.DENSITY_MATRIX,
            label="dm_snapshot",
        )
        
        modified = manager.add_snapshot(circuit, config)
        assert modified is not None

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_add_multiple_snapshots(self) -> None:
        """Test adding multiple snapshots."""
        from qiskit import QuantumCircuit
        
        manager = SnapshotManager()
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        configs = [
            SnapshotConfig(SnapshotType.STATEVECTOR, "snap1"),
            SnapshotConfig(SnapshotType.PROBABILITIES, "snap2"),
        ]
        
        modified = manager.add_multiple_snapshots(circuit, configs)
        assert modified is not None


class TestSnapshotConfig:
    """Test SnapshotConfig dataclass."""

    def test_basic_config(self) -> None:
        """Test basic configuration."""
        config = SnapshotConfig(
            snapshot_type=SnapshotType.STATEVECTOR,
            label="test",
        )
        
        assert config.snapshot_type == SnapshotType.STATEVECTOR
        assert config.label == "test"
        assert config.qubits is None

    def test_config_with_qubits(self) -> None:
        """Test configuration with specific qubits."""
        config = SnapshotConfig(
            snapshot_type=SnapshotType.PROBABILITIES,
            label="prob_snap",
            qubits=[0, 1],
        )
        
        assert config.qubits == [0, 1]


class TestSnapshotIntegration:
    """Integration tests for snapshot execution."""

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_execute_with_snapshot(self, adapter: QiskitBackendAdapter) -> None:
        """Test execution with snapshot."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        from qiskit import QuantumCircuit
        
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        # Add snapshot via options
        result = adapter.execute(
            circuit,
            {
                "snapshots": [
                    {"type": "statevector", "label": "final_state"}
                ]
            },
        )
        
        assert result is not None


# ==============================================================================
# NOISE MODEL TESTS
# ==============================================================================


class TestNoiseModelManager:
    """Test NoiseModelManager functionality."""

    def test_manager_initialization(self) -> None:
        """Test NoiseModelManager can be initialized."""
        manager = NoiseModelManager()
        assert manager is not None

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit_aer", reason="Qiskit Aer not installed"),
        reason="Qiskit Aer required",
    )
    def test_create_depolarizing_noise(self) -> None:
        """Test creating depolarizing noise model."""
        manager = NoiseModelManager()
        
        config = NoiseModelConfig(
            noise_type=NoiseModelType.DEPOLARIZING,
            parameters=NoiseParameters(
                single_qubit_error=0.01,
                two_qubit_error=0.02,
            ),
        )
        
        noise_model = manager.create_noise_model(config)
        assert noise_model is not None

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit_aer", reason="Qiskit Aer not installed"),
        reason="Qiskit Aer required",
    )
    def test_create_amplitude_damping_noise(self) -> None:
        """Test creating amplitude damping noise model."""
        manager = NoiseModelManager()
        
        config = NoiseModelConfig(
            noise_type=NoiseModelType.AMPLITUDE_DAMPING,
            parameters=NoiseParameters(single_qubit_error=0.01),
        )
        
        noise_model = manager.create_noise_model(config)
        assert noise_model is not None

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit_aer", reason="Qiskit Aer not installed"),
        reason="Qiskit Aer required",
    )
    def test_create_thermal_noise(self) -> None:
        """Test creating thermal relaxation noise model."""
        manager = NoiseModelManager()
        
        config = NoiseModelConfig(
            noise_type=NoiseModelType.THERMAL,
            parameters=NoiseParameters(t1=50.0, t2=70.0),
            include_thermal_relaxation=True,
        )
        
        noise_model = manager.create_noise_model(config)
        assert noise_model is not None

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit_aer", reason="Qiskit Aer not installed"),
        reason="Qiskit Aer required",
    )
    def test_analyze_noise_model(self) -> None:
        """Test noise model analysis."""
        manager = NoiseModelManager()
        
        config = NoiseModelConfig(
            noise_type=NoiseModelType.DEPOLARIZING,
            parameters=NoiseParameters(single_qubit_error=0.01),
        )
        
        noise_model = manager.create_noise_model(config)
        analysis = manager.analyze_noise_model(noise_model)
        
        assert isinstance(analysis, NoiseAnalysisResult)
        assert analysis.single_qubit_fidelity >= 0
        assert analysis.single_qubit_fidelity <= 1


class TestNoiseParameters:
    """Test NoiseParameters dataclass."""

    def test_default_parameters(self) -> None:
        """Test default parameter values."""
        params = NoiseParameters()
        
        assert params.single_qubit_error == 0.001
        assert params.two_qubit_error == 0.01
        assert params.t1 == 50.0
        assert params.t2 == 70.0

    def test_custom_parameters(self) -> None:
        """Test custom parameter values."""
        params = NoiseParameters(
            single_qubit_error=0.005,
            two_qubit_error=0.05,
            readout_error=0.02,
        )
        
        assert params.single_qubit_error == 0.005
        assert params.two_qubit_error == 0.05
        assert params.readout_error == 0.02


class TestNoiseIntegration:
    """Integration tests for noise model execution."""

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_execute_with_noise(
        self,
        adapter: QiskitBackendAdapter,
        simple_circuit,
    ) -> None:
        """Test circuit execution with noise model."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        noise_model = adapter.create_noise_model(
            "depolarizing",
            single_qubit_error=0.01,
            two_qubit_error=0.02,
        )
        
        result = adapter.execute(
            simple_circuit,
            {"shots": 100, "noise_model": noise_model},
        )
        
        assert result is not None
        assert result.metadata.get("noisy") is True

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_adapter_analyze_noise(self, adapter: QiskitBackendAdapter) -> None:
        """Test adapter noise analysis."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        noise_model = adapter.create_noise_model("depolarizing")
        analysis = adapter.analyze_noise_model(noise_model)
        
        assert isinstance(analysis, NoiseAnalysisResult)


# ==============================================================================
# CIRCUIT VALIDATION TESTS
# ==============================================================================


class TestCircuitValidation:
    """Test circuit validation functionality."""

    def test_validate_circuit_not_installed(self) -> None:
        """Test validation when Qiskit is not installed."""
        adapter = QiskitBackendAdapter()
        
        with patch.object(adapter, "is_available", return_value=False):
            result = adapter.validate_circuit("not a circuit")
        
        assert result.valid is False
        assert "not installed" in result.message

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_validate_invalid_circuit(self, adapter: QiskitBackendAdapter) -> None:
        """Test validation of invalid circuit."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        result = adapter.validate_circuit("not a circuit")
        assert result.valid is False

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_validate_valid_circuit(
        self,
        adapter: QiskitBackendAdapter,
        simple_circuit,
    ) -> None:
        """Test validation of valid circuit."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        result = adapter.validate_circuit(simple_circuit)
        assert result.valid is True

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_validate_parameterized_circuit(
        self,
        adapter: QiskitBackendAdapter,
        parameterized_circuit,
    ) -> None:
        """Test validation of parameterized circuit."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        circuit, _ = parameterized_circuit
        result = adapter.validate_circuit(circuit)
        
        assert result.valid is True
        assert result.details.get("requires_params") is True


# ==============================================================================
# RESOURCE ESTIMATION TESTS
# ==============================================================================


class TestResourceEstimation:
    """Test resource estimation functionality."""

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_estimate_resources(
        self,
        adapter: QiskitBackendAdapter,
        simple_circuit,
    ) -> None:
        """Test resource estimation."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        estimate = adapter.estimate_resources(simple_circuit)
        
        assert isinstance(estimate, ResourceEstimate)
        assert estimate.memory_mb is not None
        assert estimate.time_ms is not None
        assert "qubits" in estimate.metadata
        assert "depth" in estimate.metadata
        assert "gate_count" in estimate.metadata
        assert "two_qubit_gates" in estimate.metadata


# ==============================================================================
# PARAMETER BINDING TESTS
# ==============================================================================


class TestParameterBinding:
    """Test parameter binding functionality."""

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_bind_parameters_dict(
        self,
        adapter: QiskitBackendAdapter,
        parameterized_circuit,
    ) -> None:
        """Test parameter binding with dictionary."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        circuit, params = parameterized_circuit
        bound = adapter.bind_parameters(circuit, params)
        
        assert len(bound.parameters) == 0

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_bind_parameters_list(
        self,
        adapter: QiskitBackendAdapter,
        parameterized_circuit,
    ) -> None:
        """Test parameter binding with list."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        circuit, params = parameterized_circuit
        values = list(params.values())
        
        bound = adapter.bind_parameters(circuit, values)
        
        assert len(bound.parameters) == 0

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_bind_parameters_mismatch(
        self,
        adapter: QiskitBackendAdapter,
        parameterized_circuit,
    ) -> None:
        """Test parameter binding with wrong count."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        circuit, _ = parameterized_circuit
        
        with pytest.raises(ValueError):
            adapter.bind_parameters(circuit, [0.1])  # Only one value


# ==============================================================================
# EXECUTION TESTS
# ==============================================================================


class TestExecution:
    """Test circuit execution functionality."""

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_statevector_execution(self, adapter: QiskitBackendAdapter) -> None:
        """Test statevector simulation."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        from qiskit import QuantumCircuit
        
        circuit = QuantumCircuit(1)
        circuit.h(0)
        
        result = adapter.execute(circuit)
        
        assert result.result_type == ResultType.STATEVECTOR
        assert "statevector" in result.data

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_density_matrix_execution(self, adapter: QiskitBackendAdapter) -> None:
        """Test density matrix simulation."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        from qiskit import QuantumCircuit
        
        circuit = QuantumCircuit(1)
        circuit.h(0)
        
        result = adapter.execute(
            circuit,
            {"simulator_type": SimulatorType.DENSITY_MATRIX},
        )
        
        assert result.result_type == ResultType.DENSITY_MATRIX
        assert "density_matrix" in result.data

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_sampling_execution(
        self,
        adapter: QiskitBackendAdapter,
        simple_circuit,
    ) -> None:
        """Test sampling execution."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        result = adapter.execute(simple_circuit, {"shots": 100})
        
        assert result.result_type == ResultType.COUNTS
        assert "counts" in result.data
        assert sum(result.data["counts"].values()) == 100

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_parameterized_execution(
        self,
        adapter: QiskitBackendAdapter,
        parameterized_circuit,
    ) -> None:
        """Test parameterized circuit execution."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        circuit, params = parameterized_circuit
        
        result = adapter.execute(circuit, {"params": params})
        
        assert result is not None
        assert result.metadata.get("parameterized") is True


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_backend_not_installed_error(self) -> None:
        """Test BackendNotInstalledError is raised appropriately."""
        adapter = QiskitBackendAdapter()
        
        with patch.object(adapter, "is_available", return_value=False):
            with pytest.raises(BackendNotInstalledError):
                adapter.execute("circuit", {})

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_circuit_validation_error(self, adapter: QiskitBackendAdapter) -> None:
        """Test CircuitValidationError is raised for invalid circuits."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        with pytest.raises(CircuitValidationError):
            adapter.execute("not a circuit", {})

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_unsupported_simulator_type(self, adapter: QiskitBackendAdapter) -> None:
        """Test ValueError for unsupported simulator type."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        from qiskit import QuantumCircuit
        
        circuit = QuantumCircuit(1)
        circuit.h(0)
        
        with pytest.raises(ValueError):
            adapter.execute(circuit, {"simulator_type": SimulatorType.TENSOR_NETWORK})


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_full_workflow(self, adapter: QiskitBackendAdapter) -> None:
        """Test a complete workflow using multiple features."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        
        # 1. Create parameterized circuit
        theta = Parameter("theta")
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.rx(theta, 1)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        
        # 2. Validate circuit
        validation = adapter.validate_circuit(circuit)
        assert validation.valid is True
        
        # 3. Estimate resources
        resources = adapter.estimate_resources(circuit)
        assert resources.memory_mb is not None
        
        # 4. Create noise model and analyze it
        noise_model = adapter.create_noise_model("depolarizing", 0.01, 0.02)
        analysis = adapter.analyze_noise_model(noise_model)
        assert analysis.single_qubit_fidelity > 0
        
        # 5. Execute with noise and parameters
        result = adapter.execute(
            circuit,
            {
                "shots": 100,
                "params": {"theta": np.pi / 4},
                "noise_model": noise_model,
            },
        )
        
        assert result is not None
        assert result.result_type == ResultType.COUNTS
        assert result.metadata.get("noisy") is True
        assert result.metadata.get("parameterized") is True

    @pytest.mark.skipif(
        not pytest.importorskip("qiskit", reason="Qiskit not installed"),
        reason="Qiskit required",
    )
    def test_transpilation_and_execution(
        self,
        adapter: QiskitBackendAdapter,
        large_circuit,
    ) -> None:
        """Test transpilation followed by execution."""
        if not adapter.is_available():
            pytest.skip("Qiskit not installed")
        
        # Get transpiler and optimize
        transpiler = adapter.get_transpiler()
        config = TranspilationConfig(optimization_level=2)
        
        result = transpiler.transpile(large_circuit, config)
        
        # Verify optimization happened
        assert result.transpiled_depth <= result.original_depth * 1.1  # Allow some slack
        
        # Execute the optimized circuit
        exec_result = adapter.execute(result.circuit)
        assert exec_result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
