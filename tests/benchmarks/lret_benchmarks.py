"""
LRET Backend Performance Benchmarks

Comprehensive performance benchmarks specifically for the LRET backend adapter.
Tests execution speed, memory usage, and scalability across different circuit sizes.
"""

import gc
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass
class LRETBenchmarkConfig:
    """Configuration for LRET benchmark runs."""

    # Qubit configurations to test
    qubit_counts: list[int] = field(default_factory=lambda: [2, 4, 6, 8, 10, 12, 14, 16])

    # Number of shots per execution
    shots: int = 1024

    # Number of repetitions for timing
    repetitions: int = 5

    # Warmup runs before measurement
    warmup_runs: int = 2

    # Circuit types to benchmark
    circuit_types: list[str] = field(default_factory=lambda: [
        "bell",
        "ghz",
        "random",
        "variational",
        "hadamard_chain",
    ])

    # Timeout per benchmark (seconds)
    timeout: float = 60.0

    # Whether to use mock backend
    use_mock: bool = True

    # Whether to measure memory
    measure_memory: bool = True


@dataclass
class LRETBenchmarkResult:
    """Result of a single LRET benchmark run."""

    circuit_type: str
    num_qubits: int
    shots: int
    gate_count: int

    # Timing statistics (milliseconds)
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float

    # Additional metrics
    throughput: float  # shots/second
    memory_mb: float | None = None
    gates_per_second: float | None = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    successful: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "circuit_type": self.circuit_type,
            "num_qubits": self.num_qubits,
            "shots": self.shots,
            "gate_count": self.gate_count,
            "mean_time_ms": self.mean_time_ms,
            "std_time_ms": self.std_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "median_time_ms": self.median_time_ms,
            "throughput": self.throughput,
            "memory_mb": self.memory_mb,
            "gates_per_second": self.gates_per_second,
            "timestamp": self.timestamp,
            "successful": self.successful,
            "error_message": self.error_message,
        }


class LRETCircuitGenerator:
    """Generate test circuits for LRET benchmarking."""

    @staticmethod
    def bell_circuit(num_qubits: int) -> dict[str, Any]:
        """Generate Bell state circuit."""
        gates = [
            {"name": "H", "qubits": [0]},
        ]
        for i in range(1, num_qubits):
            gates.append({"name": "CX", "qubits": [0, i]})

        return {
            "num_qubits": num_qubits,
            "gates": gates,
        }

    @staticmethod
    def ghz_circuit(num_qubits: int) -> dict[str, Any]:
        """Generate GHZ state circuit."""
        gates = [
            {"name": "H", "qubits": [0]},
        ]
        for i in range(num_qubits - 1):
            gates.append({"name": "CX", "qubits": [i, i + 1]})

        return {
            "num_qubits": num_qubits,
            "gates": gates,
        }

    @staticmethod
    def random_circuit(num_qubits: int, depth: int = 10, seed: int | None = None) -> dict[str, Any]:
        """Generate random circuit."""
        rng = np.random.default_rng(seed)
        single_gates = ["H", "X", "Y", "Z"]
        two_qubit_gates = ["CX", "CZ"]

        gates = []
        for _ in range(depth):
            # Add single-qubit gates
            for q in range(num_qubits):
                if rng.random() > 0.5:
                    gate = rng.choice(single_gates)
                    gates.append({"name": gate, "qubits": [q]})

            # Add two-qubit gates
            if num_qubits >= 2:
                for q in range(0, num_qubits - 1, 2):
                    if rng.random() > 0.5:
                        gate = rng.choice(two_qubit_gates)
                        gates.append({"name": gate, "qubits": [q, q + 1]})

        return {
            "num_qubits": num_qubits,
            "gates": gates,
        }

    @staticmethod
    def variational_circuit(num_qubits: int, layers: int = 3) -> dict[str, Any]:
        """Generate variational (ansatz-like) circuit."""
        gates = []

        for _ in range(layers):
            # Rotation layer
            for q in range(num_qubits):
                gates.append({"name": "H", "qubits": [q]})

            # Entanglement layer
            for q in range(num_qubits - 1):
                gates.append({"name": "CX", "qubits": [q, q + 1]})

            # Final rotation
            for q in range(num_qubits):
                gates.append({"name": "X", "qubits": [q]})

        return {
            "num_qubits": num_qubits,
            "gates": gates,
        }

    @staticmethod
    def hadamard_chain(num_qubits: int) -> dict[str, Any]:
        """Generate Hadamard chain circuit."""
        gates = []
        for q in range(num_qubits):
            gates.append({"name": "H", "qubits": [q]})

        return {
            "num_qubits": num_qubits,
            "gates": gates,
        }

    @classmethod
    def generate(cls, circuit_type: str, num_qubits: int, **kwargs: Any) -> dict[str, Any]:
        """Generate circuit by type."""
        generators = {
            "bell": cls.bell_circuit,
            "ghz": cls.ghz_circuit,
            "random": cls.random_circuit,
            "variational": cls.variational_circuit,
            "hadamard_chain": cls.hadamard_chain,
        }

        generator = generators.get(circuit_type)
        if generator is None:
            raise ValueError(f"Unknown circuit type: {circuit_type}")

        return generator(num_qubits, **kwargs)


class LRETBenchmarkRunner:
    """Run LRET performance benchmarks."""

    def __init__(self, config: LRETBenchmarkConfig | None = None) -> None:
        """Initialize benchmark runner."""
        self.config = config or LRETBenchmarkConfig()
        self._adapter = None
        self._results: list[LRETBenchmarkResult] = []

    def _get_adapter(self) -> Any:
        """Get LRET backend adapter."""
        if self._adapter is None:
            from proxima.backends.lret import LRETBackendAdapter
            self._adapter = LRETBackendAdapter()
            if self.config.use_mock:
                self._adapter.use_mock_backend(True)
        return self._adapter

    def run_single_benchmark(
        self,
        circuit_type: str,
        num_qubits: int,
    ) -> LRETBenchmarkResult:
        """Run a single benchmark configuration."""
        adapter = self._get_adapter()
        generator = LRETCircuitGenerator()

        # Generate circuit
        try:
            circuit = generator.generate(circuit_type, num_qubits)
        except Exception as e:
            return LRETBenchmarkResult(
                circuit_type=circuit_type,
                num_qubits=num_qubits,
                shots=self.config.shots,
                gate_count=0,
                mean_time_ms=0,
                std_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                median_time_ms=0,
                throughput=0,
                successful=False,
                error_message=f"Circuit generation failed: {e}",
            )

        gate_count = len(circuit.get("gates", []))

        # Warmup runs
        for _ in range(self.config.warmup_runs):
            try:
                adapter.execute(circuit, shots=self.config.shots)
            except Exception:
                pass

        # Force garbage collection
        gc.collect()

        # Timed runs
        times: list[float] = []
        for _ in range(self.config.repetitions):
            start_time = time.perf_counter()
            try:
                adapter.execute(circuit, shots=self.config.shots)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                times.append(elapsed_ms)
            except Exception as e:
                return LRETBenchmarkResult(
                    circuit_type=circuit_type,
                    num_qubits=num_qubits,
                    shots=self.config.shots,
                    gate_count=gate_count,
                    mean_time_ms=0,
                    std_time_ms=0,
                    min_time_ms=0,
                    max_time_ms=0,
                    median_time_ms=0,
                    throughput=0,
                    successful=False,
                    error_message=f"Execution failed: {e}",
                )

        # Compute statistics
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)

        # Compute throughput (shots per second)
        throughput = (self.config.shots / mean_time) * 1000 if mean_time > 0 else 0

        # Compute gates per second
        gates_per_second = (gate_count / mean_time) * 1000 if mean_time > 0 and gate_count > 0 else None

        # Estimate memory usage
        memory_mb = None
        if self.config.measure_memory:
            # Rough estimate: 16 bytes per complex amplitude
            memory_mb = (2 ** num_qubits * 16) / (1024 * 1024)

        return LRETBenchmarkResult(
            circuit_type=circuit_type,
            num_qubits=num_qubits,
            shots=self.config.shots,
            gate_count=gate_count,
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            median_time_ms=median_time,
            throughput=throughput,
            memory_mb=memory_mb,
            gates_per_second=gates_per_second,
        )

    def run_all_benchmarks(self) -> list[LRETBenchmarkResult]:
        """Run all configured benchmarks."""
        self._results = []

        for circuit_type in self.config.circuit_types:
            for num_qubits in self.config.qubit_counts:
                result = self.run_single_benchmark(circuit_type, num_qubits)
                self._results.append(result)
                print(f"  {circuit_type:15} | {num_qubits:2} qubits | {result.mean_time_ms:8.2f} ms | {result.throughput:10.0f} shots/s")

        return self._results

    def run_scalability_benchmark(
        self,
        circuit_type: str = "ghz",
        max_qubits: int = 20,
    ) -> list[LRETBenchmarkResult]:
        """Run scalability benchmark for a specific circuit type."""
        results = []
        
        print(f"\nScalability Benchmark: {circuit_type}")
        print("-" * 60)
        
        for num_qubits in range(2, max_qubits + 1, 2):
            result = self.run_single_benchmark(circuit_type, num_qubits)
            results.append(result)
            
            if result.successful:
                print(f"  {num_qubits:2} qubits | {result.mean_time_ms:10.2f} ms | Memory: {result.memory_mb:.2f} MB")
            else:
                print(f"  {num_qubits:2} qubits | FAILED: {result.error_message}")
                break

        return results

    def run_throughput_benchmark(
        self,
        circuit_type: str = "bell",
        num_qubits: int = 4,
        shot_counts: list[int] | None = None,
    ) -> list[LRETBenchmarkResult]:
        """Run throughput benchmark with varying shot counts."""
        if shot_counts is None:
            shot_counts = [100, 500, 1000, 2000, 5000, 10000]

        results = []
        original_shots = self.config.shots

        print(f"\nThroughput Benchmark: {circuit_type} ({num_qubits} qubits)")
        print("-" * 60)

        for shots in shot_counts:
            self.config.shots = shots
            result = self.run_single_benchmark(circuit_type, num_qubits)
            results.append(result)

            if result.successful:
                print(f"  {shots:6} shots | {result.mean_time_ms:8.2f} ms | {result.throughput:10.0f} shots/s")
            else:
                print(f"  {shots:6} shots | FAILED: {result.error_message}")

        self.config.shots = original_shots
        return results

    def generate_report(self) -> str:
        """Generate a human-readable benchmark report."""
        if not self._results:
            return "No benchmark results available."

        lines = [
            "=" * 80,
            "LRET Backend Performance Benchmark Report",
            "=" * 80,
            f"Timestamp: {datetime.now(timezone.utc).isoformat()}",
            f"Configuration:",
            f"  - Shots per run: {self.config.shots}",
            f"  - Repetitions: {self.config.repetitions}",
            f"  - Warmup runs: {self.config.warmup_runs}",
            f"  - Using mock backend: {self.config.use_mock}",
            "",
            "-" * 80,
            "Results by Circuit Type:",
            "-" * 80,
        ]

        # Group by circuit type
        by_type: dict[str, list[LRETBenchmarkResult]] = {}
        for result in self._results:
            if result.circuit_type not in by_type:
                by_type[result.circuit_type] = []
            by_type[result.circuit_type].append(result)

        for circuit_type, results in by_type.items():
            lines.append(f"\n{circuit_type.upper()}:")
            lines.append(f"{'Qubits':<8} {'Gates':<8} {'Mean (ms)':<12} {'Std (ms)':<10} {'Throughput (shots/s)':<20}")
            lines.append("-" * 60)

            for r in sorted(results, key=lambda x: x.num_qubits):
                if r.successful:
                    lines.append(
                        f"{r.num_qubits:<8} {r.gate_count:<8} {r.mean_time_ms:<12.2f} "
                        f"{r.std_time_ms:<10.2f} {r.throughput:<20.0f}"
                    )
                else:
                    lines.append(f"{r.num_qubits:<8} FAILED: {r.error_message}")

        # Summary statistics
        successful_results = [r for r in self._results if r.successful]
        if successful_results:
            avg_throughput = statistics.mean(r.throughput for r in successful_results)
            total_gates = sum(r.gate_count for r in successful_results)
            
            lines.extend([
                "",
                "-" * 80,
                "Summary:",
                "-" * 80,
                f"Total benchmarks run: {len(self._results)}",
                f"Successful: {len(successful_results)}",
                f"Average throughput: {avg_throughput:.0f} shots/second",
                f"Total gates simulated: {total_gates}",
            ])

        lines.append("=" * 80)
        return "\n".join(lines)


def run_lret_benchmarks(
    use_mock: bool = True,
    quick: bool = False,
) -> list[LRETBenchmarkResult]:
    """Run LRET benchmarks with default configuration.
    
    Args:
        use_mock: Whether to use mock backend (True) or real LRET (False)
        quick: If True, run a quick benchmark with fewer configurations
        
    Returns:
        List of benchmark results
    """
    if quick:
        config = LRETBenchmarkConfig(
            qubit_counts=[2, 4, 6, 8],
            circuit_types=["bell", "ghz"],
            repetitions=3,
            warmup_runs=1,
            use_mock=use_mock,
        )
    else:
        config = LRETBenchmarkConfig(use_mock=use_mock)

    runner = LRETBenchmarkRunner(config)

    print("=" * 80)
    print("LRET Backend Performance Benchmarks")
    print("=" * 80)
    print(f"Using mock backend: {use_mock}")
    print(f"Shots per run: {config.shots}")
    print("-" * 80)

    results = runner.run_all_benchmarks()

    print("\n")
    print(runner.generate_report())

    return results


def benchmark_api_verification() -> dict[str, Any]:
    """Benchmark the API verification process."""
    from proxima.backends.lret import LRETAPIVerifier

    times = []
    for _ in range(10):
        start = time.perf_counter()
        verifier = LRETAPIVerifier(None)
        verifier.verify()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return {
        "operation": "API Verification",
        "mean_time_ms": statistics.mean(times),
        "std_time_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_time_ms": min(times),
        "max_time_ms": max(times),
    }


def benchmark_result_normalization() -> dict[str, Any]:
    """Benchmark the result normalization process."""
    from proxima.backends.lret import LRETResultNormalizer

    # Create test data
    test_counts = {"00": 500, "01": 250, "10": 150, "11": 100}
    test_statevector = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)

    times = []
    for _ in range(100):
        normalizer = LRETResultNormalizer(2)
        
        start = time.perf_counter()
        normalizer.normalize(test_counts)
        normalizer.normalize(test_statevector)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return {
        "operation": "Result Normalization (2 formats)",
        "mean_time_ms": statistics.mean(times),
        "std_time_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_time_ms": min(times),
        "max_time_ms": max(times),
    }


if __name__ == "__main__":
    # Run quick benchmarks by default
    results = run_lret_benchmarks(use_mock=True, quick=True)
    
    print("\n" + "=" * 80)
    print("Additional Benchmarks")
    print("=" * 80)
    
    # API verification benchmark
    api_result = benchmark_api_verification()
    print(f"\n{api_result['operation']}:")
    print(f"  Mean: {api_result['mean_time_ms']:.3f} ms")
    
    # Result normalization benchmark
    norm_result = benchmark_result_normalization()
    print(f"\n{norm_result['operation']}:")
    print(f"  Mean: {norm_result['mean_time_ms']:.3f} ms")
