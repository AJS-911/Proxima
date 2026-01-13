"""Cirq backend adapter (DensityMatrix + StateVector) with comprehensive error handling.

Implements Step 1.1.3c: Cirq adapter with:
- StateVector simulation
- DensityMatrix simulation
- Noise model support
- Moment optimization
- Parameter resolution for variational circuits
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


class CirqBackendAdapter(BaseBackendAdapter):
    """Cirq backend adapter with advanced simulation features.
    
    Supports:
    - State vector and density matrix simulation
    - Noise models (depolarizing, bit-flip, amplitude damping)
    - Parameter resolution for variational circuits
    - Moment optimization for faster execution
    - Expectation value computation
    """
    
    def __init__(self) -> None:
        """Initialize the Cirq adapter."""
        self._cirq: Any = None
        self._cached_version: str | None = None
        
    def get_name(self) -> str:
        return "cirq"

    def get_version(self) -> str:
        if self._cached_version:
            return self._cached_version
            
        spec = importlib.util.find_spec("cirq")
        if spec and spec.loader:
            try:
                import cirq

                self._cached_version = getattr(cirq, "__version__", "unknown")
                return self._cached_version
            except Exception:
                return "unknown"
        return "unavailable"

    def is_available(self) -> bool:
        return importlib.util.find_spec("cirq") is not None
        
    def _get_cirq(self) -> Any:
        """Get cirq module, caching for performance."""
        if self._cirq is None:
            if not self.is_available():
                raise BackendNotInstalledError("cirq", ["cirq"])
            import cirq
            self._cirq = cirq
        return self._cirq

    def get_capabilities(self) -> Capabilities:
        return Capabilities(
            simulator_types=[SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX],
            max_qubits=30,
            supports_noise=True,
            supports_gpu=False,
            supports_batching=True,  # Supports batch parameter sweeps
            custom_features={
                "parameter_resolution": True,
                "expectation_values": True,
                "moment_optimization": True,
                "noise_models": ["depolarizing", "bit_flip", "amplitude_damping", "phase_damping"],
            },
        )

    def validate_circuit(self, circuit: Any) -> ValidationResult:
        if not self.is_available():
            return ValidationResult(valid=False, message="cirq not installed")
        try:
            cirq = self._get_cirq()
        except Exception as exc:  # pragma: no cover - defensive
            return ValidationResult(valid=False, message=f"cirq import failed: {exc}")

        if not isinstance(circuit, getattr(cirq, "Circuit", ())):
            return ValidationResult(valid=False, message="input is not a cirq.Circuit")
            
        # Check for unresolved parameters
        if hasattr(circuit, "has_uncomputed_moments"):
            if circuit.has_uncomputed_moments():
                return ValidationResult(
                    valid=True,
                    message="Circuit has parameterized gates - requires parameter resolution",
                    details={"requires_params": True},
                )
        
        return ValidationResult(valid=True, message="ok")

    def estimate_resources(self, circuit: Any) -> ResourceEstimate:
        if not self.is_available():
            return ResourceEstimate(
                memory_mb=None, time_ms=None, metadata={"reason": "cirq not installed"}
            )
        try:
            cirq = self._get_cirq()
        except Exception:
            return ResourceEstimate(
                memory_mb=None, time_ms=None, metadata={"reason": "cirq import failed"}
            )

        if not isinstance(circuit, getattr(cirq, "Circuit", ())):
            return ResourceEstimate(
                memory_mb=None, time_ms=None, metadata={"reason": "not a cirq.Circuit"}
            )

        qubits = len(circuit.all_qubits())
        gate_count = sum(len(m) for m in circuit)
        depth = len(circuit)  # Number of moments = depth
        # Estimate memory: statevector needs 2^n * 16 bytes (complex128)
        memory_mb = (2**qubits * 16) / (1024 * 1024) if qubits <= 30 else None
        
        # Estimate time based on gate count and depth
        time_ms = gate_count * 0.01 + depth * 0.1 if gate_count > 0 else None
        
        metadata = {
            "qubits": qubits,
            "gate_count": gate_count,
            "depth": depth,
            "two_qubit_gates": self._count_two_qubit_gates(circuit),
        }
        return ResourceEstimate(memory_mb=memory_mb, time_ms=time_ms, metadata=metadata)
        
    def _count_two_qubit_gates(self, circuit: Any) -> int:
        """Count two-qubit gates in circuit."""
        count = 0
        for moment in circuit:
            for op in moment:
                if len(op.qubits) >= 2:
                    count += 1
        return count
        
    def optimize_circuit(self, circuit: Any, level: int = 1) -> Any:
        """Optimize circuit using Cirq's optimization passes.
        
        Args:
            circuit: Cirq circuit to optimize
            level: Optimization level (0=none, 1=basic, 2=aggressive)
            
        Returns:
            Optimized circuit
        """
        if not self.is_available():
            return circuit
            
        try:
            cirq = self._get_cirq()
            
            if level == 0:
                return circuit
                
            optimized = circuit.copy()
            
            if level >= 1:
                # Basic optimizations
                optimized = cirq.drop_empty_moments(optimized)
                optimized = cirq.drop_negligible_operations(optimized)
                
            if level >= 2:
                # Aggressive optimizations
                try:
                    optimized = cirq.merge_single_qubit_gates_to_phased_x_and_z(optimized)
                except Exception as e:
                    logger.debug(f"Advanced optimization failed: {e}")
                    
            return optimized
            
        except Exception as e:
            logger.warning(f"Circuit optimization failed: {e}")
            return circuit
            
    def create_noise_model(
        self,
        noise_type: str = "depolarizing",
        p: float = 0.01,
        **kwargs: Any,
    ) -> Any:
        """Create a noise model for noisy simulation.
        
        Args:
            noise_type: Type of noise ("depolarizing", "bit_flip", "amplitude_damping", "phase_damping")
            p: Error probability
            **kwargs: Additional parameters for specific noise types
            
        Returns:
            Cirq NoiseModel or list of noise channels
        """
        if not self.is_available():
            raise BackendNotInstalledError("cirq", ["cirq"])
            
        cirq = self._get_cirq()
        
        if noise_type == "depolarizing":
            return cirq.ConstantQubitNoiseModel(cirq.depolarize(p))
        elif noise_type == "bit_flip":
            return cirq.ConstantQubitNoiseModel(cirq.bit_flip(p))
        elif noise_type == "amplitude_damping":
            return cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(p))
        elif noise_type == "phase_damping":
            return cirq.ConstantQubitNoiseModel(cirq.phase_damp(p))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
            
    def resolve_parameters(
        self,
        circuit: Any,
        params: dict[str, float],
    ) -> Any:
        """Resolve parameter symbols in a variational circuit.
        
        Args:
            circuit: Parameterized Cirq circuit
            params: Dict mapping parameter names to values
            
        Returns:
            Resolved circuit with concrete values
        """
        if not self.is_available():
            raise BackendNotInstalledError("cirq", ["cirq"])
            
        cirq = self._get_cirq()
        
        # Build resolver from params dict
        resolver = cirq.ParamResolver(params)
        
        return cirq.resolve_parameters(circuit, resolver)
        
    def compute_expectation(
        self,
        circuit: Any,
        observable: Any,
        params: dict[str, float] | None = None,
    ) -> float:
        """Compute expectation value of an observable.
        
        Args:
            circuit: Cirq circuit
            observable: Pauli observable or PauliSum
            params: Optional parameter values for variational circuits
            
        Returns:
            Expectation value as float
        """
        if not self.is_available():
            raise BackendNotInstalledError("cirq", ["cirq"])
            
        cirq = self._get_cirq()
        
        resolved_circuit = circuit
        if params:
            resolved_circuit = self.resolve_parameters(circuit, params)
            
        simulator = cirq.Simulator()
        result = simulator.simulate_expectation_values(
            resolved_circuit,
            observables=[observable],
        )
        
        return float(result[0].real)

    def execute(
        self, circuit: Any, options: dict[str, Any] | None = None
    ) -> ExecutionResult:
        if not self.is_available():
            raise BackendNotInstalledError("cirq", ["cirq"])

        # Validate circuit first
        validation = self.validate_circuit(circuit)
        if not validation.valid:
            raise CircuitValidationError(
                backend_name="cirq",
                reason=validation.message or "Invalid circuit",
            )

        try:
            cirq = self._get_cirq()
        except ImportError as exc:
            raise BackendNotInstalledError("cirq", ["cirq"], original_exception=exc)

        options = options or {}
        sim_type = options.get("simulator_type", SimulatorType.STATE_VECTOR)
        repetitions = int(options.get("repetitions", options.get("shots", 0)))
        noise_model = options.get("noise_model")
        params = options.get("params", options.get("parameters"))
        optimize_level = int(options.get("optimize", 0))

        # Check qubit limits
        qubit_count = len(circuit.all_qubits()) if hasattr(circuit, "all_qubits") else 0
        max_qubits = self.get_capabilities().max_qubits
        if qubit_count > max_qubits:
            raise QubitLimitExceededError(
                backend_name="cirq",
                requested_qubits=qubit_count,
                max_qubits=max_qubits,
            )

        if sim_type not in (SimulatorType.STATE_VECTOR, SimulatorType.DENSITY_MATRIX):
            raise ValueError(f"Unsupported simulator type: {sim_type}")

        try:
            # Resolve parameters if provided
            exec_circuit = circuit
            if params:
                exec_circuit = self.resolve_parameters(circuit, params)
                
            # Optimize if requested
            if optimize_level > 0:
                exec_circuit = self.optimize_circuit(exec_circuit, level=optimize_level)
            
            simulator: Any
            if sim_type == SimulatorType.DENSITY_MATRIX or noise_model:
                simulator = cirq.DensityMatrixSimulator(noise=noise_model)
            else:
                simulator = cirq.Simulator()

            start = time.perf_counter()
            result_type: ResultType
            data: dict[str, Any]
            raw_result: Any

            if repetitions > 0:
                raw_result = simulator.run(exec_circuit, repetitions=repetitions)
                result_type = ResultType.COUNTS
                counts: dict[str, int] = {}
                measurement_keys = list(raw_result.measurements.keys())
                if measurement_keys:
                    for key in measurement_keys:
                        histogram = raw_result.histogram(key=key)
                        for state_int, count in histogram.items():
                            n_bits = raw_result.measurements[key].shape[1]
                            bitstring = format(state_int, f"0{n_bits}b")
                            counts[bitstring] = counts.get(bitstring, 0) + count
                data = {"counts": counts, "repetitions": repetitions}
            else:
                if sim_type == SimulatorType.DENSITY_MATRIX or noise_model:
                    raw_result = simulator.simulate(exec_circuit)
                    density_matrix = raw_result.final_density_matrix
                    result_type = ResultType.DENSITY_MATRIX
                    data = {"density_matrix": density_matrix}
                else:
                    raw_result = simulator.simulate(exec_circuit)
                    statevector = raw_result.final_state_vector
                    result_type = ResultType.STATEVECTOR
                    data = {"statevector": statevector}

            execution_time_ms = (time.perf_counter() - start) * 1000.0

            return ExecutionResult(
                backend=self.get_name(),
                simulator_type=sim_type,
                execution_time_ms=execution_time_ms,
                qubit_count=qubit_count,
                shot_count=repetitions if repetitions > 0 else None,
                result_type=result_type,
                data=data,
                metadata={
                    "cirq_version": self.get_version(),
                    "optimized": optimize_level > 0,
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
            raise wrap_backend_exception(exc, "cirq", "execution")

    def supports_simulator(self, sim_type: SimulatorType) -> bool:
        return sim_type in self.get_capabilities().simulator_types
        
    def get_supported_gates(self) -> list[str]:
        """Get list of gates supported by Cirq."""
        return [
            "H", "X", "Y", "Z", "S", "T",
            "Rx", "Ry", "Rz",
            "CNOT", "CZ", "SWAP", "ISWAP",
            "CCX", "CCZ", "CSWAP",
            "PhasedXPowGate", "PhasedXZGate",
            "FSim", "XX", "YY", "ZZ",
        ]
