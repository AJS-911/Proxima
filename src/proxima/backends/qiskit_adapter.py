"""Qiskit backend adapter for Statevector and DensityMatrix simulation with edge cases.

Implements Step 1.1.3b: Qiskit Aer adapter with:
- StateVector simulation
- DensityMatrix simulation
- Noise model support
- Transpilation optimization
- Snapshot modes for intermediate state inspection
"""

from __future__ import annotations

import importlib.util
import logging
import time
from typing import Any

from proxima.backends.base import (
    BaseBackendAdapter,
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
    wrap_backend_exception,
)

logger = logging.getLogger(__name__)


class QiskitBackendAdapter(BaseBackendAdapter):
    """Qiskit Aer backend adapter with comprehensive simulation support.
    
    Supports:
    - State vector and density matrix simulation
    - Noise models from qiskit_aer
    - Transpilation with optimization levels
    - Snapshot modes for intermediate states
    - Parameter binding for variational circuits
    """
    
    def __init__(self) -> None:
        """Initialize the Qiskit adapter."""
        self._cached_version: str | None = None
        
    def get_name(self) -> str:
        return "qiskit"

    def get_version(self) -> str:
        if self._cached_version:
            return self._cached_version
            
        spec = importlib.util.find_spec("qiskit")
        if spec and spec.loader:
            try:
                import qiskit

                self._cached_version = getattr(qiskit, "__version__", "unknown")
                return self._cached_version
            except Exception:
                return "unknown"
        return "unavailable"

    def is_available(self) -> bool:
        return (
            importlib.util.find_spec("qiskit") is not None
            and importlib.util.find_spec("qiskit_aer") is not None
        )

    def get_capabilities(self) -> Capabilities:
        return Capabilities(
            simulator_types=[SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX],
            max_qubits=28,
            supports_noise=True,
            supports_gpu=False,
            supports_batching=True,  # Supports parameter binding for batched execution
            custom_features={
                "transpilation": True,
                "optimization_levels": [0, 1, 2, 3],
                "snapshot_modes": ["statevector", "density_matrix", "probabilities", "expectation_value"],
                "noise_models": True,
                "parameter_binding": True,
            },
        )

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        if not self.is_available():
            return ValidationResult(
                valid=False, message="qiskit/qiskit-aer not installed"
            )
        try:
            from qiskit import QuantumCircuit
        except Exception as exc:  # pragma: no cover - defensive
            return ValidationResult(valid=False, message=f"qiskit import failed: {exc}")

        if not isinstance(circuit, QuantumCircuit):
            return ValidationResult(
                valid=False, message="input is not a qiskit.QuantumCircuit"
            )
            
        # Check for unbound parameters
        if circuit.parameters:
            return ValidationResult(
                valid=True,
                message="Circuit has unbound parameters - requires parameter binding",
                details={
                    "requires_params": True,
                    "parameters": [str(p) for p in circuit.parameters],
                },
            )
            
        return ValidationResult(valid=True, message="ok")

    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        if not self.is_available():
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "qiskit/qiskit-aer not installed"},
            )
        try:
            from qiskit import QuantumCircuit
        except Exception:
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "qiskit import failed"},
            )

        if not isinstance(circuit, QuantumCircuit):
            return ResourceEstimate(
                memory_mb=None,
                time_ms=None,
                metadata={"reason": "not a QuantumCircuit"},
            )

        qubits = circuit.num_qubits
        depth = circuit.depth() or 0
        gate_count = circuit.size()
        # Estimate memory: statevector needs 2^n * 16 bytes (complex128)
        memory_mb = (2**qubits * 16) / (1024 * 1024) if qubits <= 28 else None
        
        # Estimate time based on gate count
        time_ms = gate_count * 0.01 + depth * 0.1 if gate_count > 0 else None
        
        # Count two-qubit gates for more accurate estimation
        two_qubit_gates = sum(
            1 for instr, _, _ in circuit.data
            if hasattr(instr, 'num_qubits') and instr.num_qubits >= 2
        )
        
        metadata = {
            "qubits": qubits,
            "depth": depth,
            "gate_count": gate_count,
            "two_qubit_gates": two_qubit_gates,
        }
        return ResourceEstimate(memory_mb=memory_mb, time_ms=time_ms, metadata=metadata)
        
    def create_noise_model(
        self,
        error_type: str = "depolarizing",
        single_qubit_error: float = 0.001,
        two_qubit_error: float = 0.01,
        **kwargs: Any,
    ) -> Any:
        """Create a noise model for noisy simulation.
        
        Args:
            error_type: Type of error ("depolarizing", "thermal", "readout")
            single_qubit_error: Error rate for single-qubit gates
            two_qubit_error: Error rate for two-qubit gates
            **kwargs: Additional parameters
            
        Returns:
            qiskit_aer NoiseModel
        """
        if not self.is_available():
            raise BackendNotInstalledError("qiskit", ["qiskit", "qiskit-aer"])
            
        from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
        
        noise_model = NoiseModel()
        
        if error_type == "depolarizing":
            # Single-qubit depolarizing error
            error_1q = depolarizing_error(single_qubit_error, 1)
            noise_model.add_all_qubit_quantum_error(
                error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'h', 'x', 'y', 'z', 's', 't']
            )
            
            # Two-qubit depolarizing error
            error_2q = depolarizing_error(two_qubit_error, 2)
            noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'swap'])
            
        elif error_type == "thermal":
            t1 = kwargs.get("t1", 50e3)  # 50 microseconds
            t2 = kwargs.get("t2", 70e3)  # 70 microseconds
            gate_time_1q = kwargs.get("gate_time_1q", 50)  # 50 ns
            gate_time_2q = kwargs.get("gate_time_2q", 300)  # 300 ns
            
            error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
            noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'h', 'x'])
            
            error_2q = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(
                thermal_relaxation_error(t1, t2, gate_time_2q)
            )
            noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
            
        return noise_model
        
    def add_snapshot(
        self,
        circuit: Any,
        snapshot_type: str = "statevector",
        label: str = "snapshot",
        qubits: list[int] | None = None,
    ) -> Any:
        """Add a snapshot instruction to the circuit.
        
        Snapshot modes allow capturing intermediate states during simulation.
        
        Args:
            circuit: Qiskit QuantumCircuit
            snapshot_type: Type of snapshot ("statevector", "density_matrix", 
                          "probabilities", "expectation_value")
            label: Label for the snapshot
            qubits: Which qubits to snapshot (None for all)
            
        Returns:
            Modified circuit with snapshot instruction
        """
        if not self.is_available():
            raise BackendNotInstalledError("qiskit", ["qiskit", "qiskit-aer"])
            
        from qiskit import QuantumCircuit
        
        if not isinstance(circuit, QuantumCircuit):
            raise CircuitValidationError(
                backend_name="qiskit",
                reason="Expected qiskit.QuantumCircuit",
            )
            
        # Create a copy to avoid modifying original
        modified = circuit.copy()
        
        if snapshot_type == "statevector":
            modified.save_statevector(label=label)
        elif snapshot_type == "density_matrix":
            modified.save_density_matrix(label=label)
        elif snapshot_type == "probabilities":
            if qubits:
                modified.save_probabilities(qubits, label=label)
            else:
                modified.save_probabilities(label=label)
        elif snapshot_type == "expectation_value":
            logger.warning("expectation_value snapshot requires observable parameter")
        else:
            raise ValueError(f"Unknown snapshot type: {snapshot_type}")
            
        return modified
        
    def bind_parameters(
        self,
        circuit: Any,
        params: dict[str, float] | list[float],
    ) -> Any:
        """Bind parameters to a variational circuit.
        
        Args:
            circuit: Parameterized QuantumCircuit
            params: Dict mapping parameter names to values, or list of values
            
        Returns:
            Circuit with bound parameters
        """
        if not self.is_available():
            raise BackendNotInstalledError("qiskit", ["qiskit", "qiskit-aer"])
            
        from qiskit import QuantumCircuit
        
        if not isinstance(circuit, QuantumCircuit):
            raise CircuitValidationError(
                backend_name="qiskit",
                reason="Expected qiskit.QuantumCircuit",
            )
            
        if isinstance(params, list):
            # Bind by order
            if len(params) != len(circuit.parameters):
                raise ValueError(
                    f"Parameter count mismatch: {len(params)} provided, "
                    f"{len(circuit.parameters)} required"
                )
            param_dict = dict(zip(circuit.parameters, params))
        else:
            # Bind by name
            param_dict = {
                p: params[str(p)]
                for p in circuit.parameters
                if str(p) in params
            }
            
        return circuit.bind_parameters(param_dict)

    def execute(
        self, circuit: Any, options: dict[str, Any] | None = None
    ) -> ExecutionResult:
        if not self.is_available():
            raise BackendNotInstalledError("qiskit", ["qiskit", "qiskit-aer"])

        # Validate circuit first
        validation = self.validate_circuit(circuit)
        if not validation.valid:
            raise CircuitValidationError(
                backend_name="qiskit",
                reason=validation.message or "Invalid circuit",
            )

        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit_aer import AerSimulator
        except ImportError as exc:
            raise BackendNotInstalledError(
                "qiskit", ["qiskit", "qiskit-aer"], original_exception=exc
            )

        if not isinstance(circuit, QuantumCircuit):
            raise CircuitValidationError(
                backend_name="qiskit",
                reason="Expected qiskit.QuantumCircuit",
                circuit_info={"type": type(circuit).__name__},
            )

        # Check qubit limits
        qubit_count = circuit.num_qubits
        max_qubits = self.get_capabilities().max_qubits
        if qubit_count > max_qubits:
            raise QubitLimitExceededError(
                backend_name="qiskit",
                requested_qubits=qubit_count,
                max_qubits=max_qubits,
            )

        options = options or {}
        sim_type = options.get("simulator_type", SimulatorType.STATE_VECTOR)
        shots = int(options.get("shots", options.get("repetitions", 0)))
        density_mode = sim_type == SimulatorType.DENSITY_MATRIX
        noise_model = options.get("noise_model")
        optimization_level = int(options.get("optimization_level", 1))
        params = options.get("params", options.get("parameters"))
        snapshots = options.get("snapshots", [])

        if sim_type not in (SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX):
            raise ValueError(f"Unsupported simulator type: {sim_type}")

        try:
            # Bind parameters if provided
            exec_circuit = circuit
            if params:
                exec_circuit = self.bind_parameters(circuit, params)
                
            method = "density_matrix" if density_mode else "statevector"
            simulator = AerSimulator(method=method, noise_model=noise_model)
            t_circuit = transpile(
                exec_circuit, 
                simulator,
                optimization_level=optimization_level,
            )

            final_circuit = t_circuit.copy()
            
            # Add snapshots if requested
            for snap in snapshots:
                if isinstance(snap, str):
                    final_circuit = self.add_snapshot(final_circuit, snap)
                elif isinstance(snap, dict):
                    final_circuit = self.add_snapshot(
                        final_circuit,
                        snap.get("type", "statevector"),
                        snap.get("label", "snapshot"),
                        snap.get("qubits"),
                    )
            
            if shots == 0:
                if density_mode:
                    final_circuit.save_density_matrix()
                else:
                    final_circuit.save_statevector()

            start = time.perf_counter()
            result = simulator.run(
                final_circuit, shots=shots if shots > 0 else None
            ).result()
            execution_time_ms = (time.perf_counter() - start) * 1000.0

            if shots > 0:
                counts = result.get_counts(final_circuit)
                data = {"counts": counts, "shots": shots}
                result_type = ResultType.COUNTS
                raw_result = result
            else:
                result_data = result.data(final_circuit)
                if density_mode:
                    density_matrix = result_data.get("density_matrix")
                    result_type = ResultType.DENSITY_MATRIX
                    data = {"density_matrix": density_matrix}
                    raw_result = result
                else:
                    statevector = result_data.get("statevector")
                    result_type = ResultType.STATEVECTOR
                    data = {"statevector": statevector}
                    raw_result = result
                    
            # Add any snapshot data
            for key in result_data.keys():
                if key.startswith("snapshot_"):
                    data[key] = result_data[key]

            return ExecutionResult(
                backend=self.get_name(),
                simulator_type=sim_type,
                execution_time_ms=execution_time_ms,
                qubit_count=qubit_count,
                shot_count=shots if shots > 0 else None,
                result_type=result_type,
                data=data,
                metadata={
                    "qiskit_version": self.get_version(),
                    "optimization_level": optimization_level,
                    "noisy": noise_model is not None,
                    "parameterized": params is not None,
                },
                raw_result=raw_result,
            )

        except (
            BackendNotInstalledError,
            CircuitValidationError,
            QubitLimitExceededError,
        ):
            raise
        except Exception as exc:
            raise wrap_backend_exception(exc, "qiskit", "execution")

    def supports_simulator(self, sim_type: SimulatorType) -> bool:
        return sim_type in self.get_capabilities().simulator_types
        
    def get_supported_gates(self) -> list[str]:
        """Get list of gates supported by Qiskit."""
        return [
            "h", "x", "y", "z", "s", "sdg", "t", "tdg",
            "rx", "ry", "rz", "u", "u1", "u2", "u3",
            "cx", "cy", "cz", "swap", "iswap",
            "ccx", "cswap",
            "reset", "measure", "barrier",
        ]
