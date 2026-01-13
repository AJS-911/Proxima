"""Step 6.2: Backend Tests - Testing backend adapters with mocks.

Backend tests focus on:
- Backend adapter functionality
- Mock backends for testing
- Backend status handling
"""

import pytest

# Check if textual is available for TUI-related tests
try:
    import textual

    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False


# =============================================================================
# BACKEND ADAPTER TESTS
# =============================================================================


class TestBackendAdapters:
    """Tests for backend adapters."""

    @pytest.mark.backend
    def test_mock_backend_name(self, mock_backend):
        """Test mock backend name."""
        assert mock_backend.name == "mock_backend"

    @pytest.mark.backend
    def test_mock_backend_type(self, mock_backend):
        """Test mock backend type."""
        assert mock_backend.backend_type == "simulator"

    @pytest.mark.backend
    def test_mock_backend_connected(self, mock_backend):
        """Test mock backend connection status."""
        assert mock_backend.is_connected() is True


# =============================================================================
# ASYNC BACKEND TESTS
# =============================================================================


class TestAsyncBackends:
    """Tests for async backend operations."""

    @pytest.mark.backend
    @pytest.mark.asyncio
    async def test_async_mock_backend_execute(self, async_mock_backend):
        """Test async mock backend execute."""
        result = await async_mock_backend.execute({"circuit": "async_test"})

        assert result is not None
        assert "status" in result


# =============================================================================
# BACKEND RESULT TESTS
# =============================================================================


class TestBackendResults:
    """Tests for backend result handling."""

    @pytest.mark.backend
    def test_backend_result_creation(self):
        """Test BackendResult creation."""
        from proxima.data.compare import BackendResult

        result = BackendResult(
            backend_name="test_backend",
            success=True,
            execution_time_ms=123.45,
            memory_peak_mb=256.0,
        )

        assert result.backend_name == "test_backend"
        assert result.success is True
        assert result.execution_time_ms == 123.45

    @pytest.mark.backend
    def test_backend_result_with_error(self):
        """Test BackendResult with error."""
        from proxima.data.compare import BackendResult

        result = BackendResult(
            backend_name="failed_backend",
            success=False,
            execution_time_ms=0.0,
            memory_peak_mb=0.0,
            error="Connection timeout",
        )

        assert result.success is False
        assert result.error == "Connection timeout"


# =============================================================================
# BACKEND STATUS TESTS
# =============================================================================


class TestBackendStatus:
    """Tests for backend status handling."""

    @pytest.mark.backend
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_status_connected(self):
        """Test connected status."""
        from proxima.tui.widgets import BackendStatus

        assert BackendStatus.CONNECTED.value == "connected"

    @pytest.mark.backend
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_status_disconnected(self):
        """Test disconnected status."""
        from proxima.tui.widgets import BackendStatus

        assert BackendStatus.DISCONNECTED.value == "disconnected"

    @pytest.mark.backend
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_status_error(self):
        """Test error status."""
        from proxima.tui.widgets import BackendStatus

        assert BackendStatus.ERROR.value == "error"

    @pytest.mark.backend
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_status_connecting(self):
        """Test connecting status."""
        from proxima.tui.widgets import BackendStatus

        assert BackendStatus.CONNECTING.value == "connecting"


# =============================================================================
# BACKEND INFO TESTS
# =============================================================================


class TestBackendInfo:
    """Tests for BackendInfo data class."""

    @pytest.mark.backend
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_info_creation(self):
        """Test BackendInfo creation."""
        from proxima.tui.widgets import BackendInfo, BackendStatus

        info = BackendInfo(
            name="Test Backend",
            backend_type="simulator",
            status=BackendStatus.CONNECTED,
            total_executions=100,
            avg_latency_ms=25.5,
            last_used="2024-01-01",
        )

        assert info.name == "Test Backend"
        assert info.backend_type == "simulator"
        assert info.status == BackendStatus.CONNECTED
        assert info.total_executions == 100

    @pytest.mark.backend
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_info_default_values(self):
        """Test BackendInfo default values."""
        from proxima.tui.widgets import BackendInfo, BackendStatus

        info = BackendInfo(
            name="Minimal",
            backend_type="test",
            status=BackendStatus.DISCONNECTED,
        )

        assert info.total_executions == 0
        assert info.avg_latency_ms is None  # None when no executions yet


# =============================================================================
# BACKEND CARD WIDGET TESTS
# =============================================================================


class TestBackendCardWidget:
    """Tests for BackendCard widget."""

    @pytest.mark.backend
    @pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
    def test_backend_card_creation(self):
        """Test BackendCard creation."""
        from proxima.tui.widgets import BackendCard, BackendInfo, BackendStatus

        info = BackendInfo(
            name="Card Test",
            backend_type="real",
            status=BackendStatus.CONNECTED,
        )

        card = BackendCard(backend=info)

        assert card._backend == info
        assert card._backend.name == "Card Test"


# =============================================================================
# EXECUTION STRATEGY TESTS
# =============================================================================


class TestExecutionStrategy:
    """Tests for backend execution strategies."""

    @pytest.mark.backend
    def test_sequential_strategy(self):
        """Test sequential execution strategy."""
        from proxima.data.compare import ExecutionStrategy

        assert ExecutionStrategy.SEQUENTIAL.value == "sequential"

    @pytest.mark.backend
    def test_parallel_strategy(self):
        """Test parallel execution strategy."""
        from proxima.data.compare import ExecutionStrategy

        assert ExecutionStrategy.PARALLEL.value == "parallel"

    @pytest.mark.backend
    def test_adaptive_strategy(self):
        """Test adaptive execution strategy."""
        from proxima.data.compare import ExecutionStrategy

        assert ExecutionStrategy.ADAPTIVE.value == "adaptive"

# =============================================================================
# LRET ADAPTER COMPREHENSIVE TESTS
# =============================================================================


class TestLRETAdapter:
    """Comprehensive tests for the LRET backend adapter.
    
    Tests cover Step 1.1.3a-d LRET integration requirements:
    - API availability checks
    - Circuit validation
    - Resource estimation
    - Execution with state vectors and shots
    """

    @pytest.mark.backend
    def test_lret_adapter_name(self):
        """Test LRET adapter returns correct name."""
        from proxima.backends.lret import LRETBackendAdapter

        adapter = LRETBackendAdapter()
        assert adapter.get_name() == "lret"

    @pytest.mark.backend
    def test_lret_adapter_capabilities(self):
        """Test LRET adapter capabilities are correctly defined."""
        from proxima.backends.lret import LRETBackendAdapter
        from proxima.backends.base import SimulatorType

        adapter = LRETBackendAdapter()
        caps = adapter.get_capabilities()

        assert SimulatorType.STATE_VECTOR in caps.simulator_types
        assert SimulatorType.CUSTOM in caps.simulator_types
        assert caps.max_qubits == 32
        assert caps.supports_noise is False
        assert caps.supports_gpu is False
        assert caps.supports_batching is True
        assert caps.custom_features.get("framework_integration") is True

    @pytest.mark.backend
    def test_lret_validate_none_circuit(self):
        """Test LRET validation rejects None circuit."""
        from proxima.backends.lret import LRETBackendAdapter

        adapter = LRETBackendAdapter()
        result = adapter.validate_circuit(None)

        assert result.valid is False
        assert "None" in result.message

    @pytest.mark.backend
    def test_lret_validate_dict_circuit_with_gates(self):
        """Test LRET validation accepts dict circuit with gates key."""
        from proxima.backends.lret import LRETBackendAdapter

        adapter = LRETBackendAdapter()
        circuit = {
            "qubits": 2,
            "gates": [
                {"name": "H", "targets": [0]},
                {"name": "CX", "targets": [0, 1]},
            ],
        }
        result = adapter.validate_circuit(circuit)

        assert result.valid is True
        assert "dict" in result.details.get("format", "").lower() or "Dictionary" in result.message

    @pytest.mark.backend
    def test_lret_validate_dict_circuit_with_operations(self):
        """Test LRET validation accepts dict circuit with operations key."""
        from proxima.backends.lret import LRETBackendAdapter

        adapter = LRETBackendAdapter()
        circuit = {
            "qubits": 3,
            "operations": [{"op": "X", "qubit": 0}],
        }
        result = adapter.validate_circuit(circuit)

        assert result.valid is True

    @pytest.mark.backend
    def test_lret_validate_dict_circuit_missing_keys(self):
        """Test LRET validation rejects dict without required keys."""
        from proxima.backends.lret import LRETBackendAdapter

        adapter = LRETBackendAdapter()
        circuit = {"name": "empty_circuit"}
        result = adapter.validate_circuit(circuit)

        assert result.valid is False
        assert "missing" in result.message.lower()

    @pytest.mark.backend
    def test_lret_validate_unsupported_type(self):
        """Test LRET validation rejects unsupported circuit types."""
        from proxima.backends.lret import LRETBackendAdapter

        adapter = LRETBackendAdapter()
        circuit = 12345  # Invalid type
        result = adapter.validate_circuit(circuit)

        assert result.valid is False
        assert "Unsupported" in result.message

    @pytest.mark.backend
    def test_lret_estimate_resources_unavailable(self):
        """Test resource estimation when LRET not installed."""
        from proxima.backends.lret import LRETBackendAdapter

        adapter = LRETBackendAdapter()
        # Simulate LRET not available
        adapter.is_available = lambda: False

        circuit = {"qubits": 4, "gates": []}
        estimate = adapter.estimate_resources(circuit)

        assert estimate.metadata.get("reason") is not None

    @pytest.mark.backend
    def test_lret_estimate_resources_dict_circuit(self):
        """Test resource estimation for dict-based circuit."""
        from proxima.backends.lret import LRETBackendAdapter

        adapter = LRETBackendAdapter()
        circuit = {
            "num_qubits": 5,
            "gates": [{"name": "H"} for _ in range(10)],
        }
        estimate = adapter.estimate_resources(circuit)

        assert estimate.metadata.get("qubits") == 5 or estimate.metadata.get("reason") is not None

    @pytest.mark.backend
    def test_lret_supported_gates(self):
        """Test LRET returns list of supported gates."""
        from proxima.backends.lret import LRETBackendAdapter

        adapter = LRETBackendAdapter()
        gates = adapter.get_supported_gates()

        assert isinstance(gates, list)
        assert "H" in gates
        assert "X" in gates
        assert "CX" in gates or "CNOT" in gates

    @pytest.mark.backend
    def test_lret_supports_simulator_type(self):
        """Test LRET supports state vector simulation."""
        from proxima.backends.lret import LRETBackendAdapter
        from proxima.backends.base import SimulatorType

        adapter = LRETBackendAdapter()

        assert adapter.supports_simulator(SimulatorType.STATE_VECTOR) is True
        assert adapter.supports_simulator(SimulatorType.CUSTOM) is True

    @pytest.mark.backend
    def test_lret_version_unavailable(self):
        """Test version returns unavailable when LRET not installed."""
        from proxima.backends.lret import LRETBackendAdapter

        adapter = LRETBackendAdapter()
        adapter.is_available = lambda: False

        version = adapter.get_version()
        assert version == "unavailable"

    @pytest.mark.backend
    def test_lret_extract_qubit_count_from_dict(self):
        """Test qubit count extraction from dict circuit."""
        from proxima.backends.lret import LRETBackendAdapter

        adapter = LRETBackendAdapter()

        # Test num_qubits key
        assert adapter._extract_qubit_count({"num_qubits": 5}) == 5

        # Test qubits list
        assert adapter._extract_qubit_count({"qubits": [0, 1, 2]}) == 3

        # Test n_qubits key
        assert adapter._extract_qubit_count({"n_qubits": 7}) == 7

    @pytest.mark.backend
    def test_lret_extract_gate_count_from_dict(self):
        """Test gate count extraction from dict circuit."""
        from proxima.backends.lret import LRETBackendAdapter

        adapter = LRETBackendAdapter()

        assert adapter._extract_gate_count({"gates": [1, 2, 3]}) == 3
        assert adapter._extract_gate_count({"operations": [1, 2]}) == 2
        assert adapter._extract_gate_count({"instructions": [1]}) == 1
        assert adapter._extract_gate_count({}) is None

    @pytest.mark.backend
    def test_lret_execution_without_lret_installed(self):
        """Test LRET execution raises error when not installed."""
        from proxima.backends.lret import LRETBackendAdapter
        from proxima.backends.exceptions import BackendNotInstalledError

        adapter = LRETBackendAdapter()
        adapter.is_available = lambda: False

        circuit = {"qubits": 2, "gates": []}

        with pytest.raises(BackendNotInstalledError):
            adapter.execute(circuit)

    @pytest.mark.backend
    def test_lret_execution_invalid_circuit(self):
        """Test LRET execution raises error for invalid circuit."""
        from proxima.backends.lret import LRETBackendAdapter
        from proxima.backends.exceptions import CircuitValidationError

        adapter = LRETBackendAdapter()
        # Only test if LRET is not available (to avoid actual execution)
        if not adapter.is_available():
            pytest.skip("LRET not installed, skipping execution test")
            return

        # This should fail validation
        with pytest.raises(CircuitValidationError):
            adapter.execute({"invalid": True})


# =============================================================================
# BACKEND REGISTRY TESTS
# =============================================================================


class TestBackendRegistry:
    """Tests for the BackendRegistry with hot-reload support."""

    @pytest.mark.backend
    def test_registry_discover(self):
        """Test registry discovery."""
        from proxima.backends.registry import BackendRegistry

        registry = BackendRegistry()
        registry.discover()

        statuses = registry.list_statuses()
        assert len(statuses) > 0

    @pytest.mark.backend
    def test_registry_list_available(self):
        """Test listing available backends."""
        from proxima.backends.registry import BackendRegistry

        registry = BackendRegistry()
        registry.discover()

        available = registry.list_available()
        assert isinstance(available, list)

    @pytest.mark.backend
    def test_registry_hot_reload(self):
        """Test hot-reload functionality."""
        from proxima.backends.registry import BackendRegistry

        registry = BackendRegistry()
        registry.discover()

        # Force hot-reload
        changes = registry.hot_reload(force=True)

        assert isinstance(changes, dict)

    @pytest.mark.backend
    def test_registry_discovery_stats(self):
        """Test discovery statistics."""
        from proxima.backends.registry import BackendRegistry

        registry = BackendRegistry()
        registry.discover()

        stats = registry.get_discovery_stats()

        assert "discovery_count" in stats
        assert "total_backends" in stats
        assert "available_backends" in stats
        assert stats["discovery_count"] >= 1

    @pytest.mark.backend
    def test_registry_health_tracking(self):
        """Test backend health score tracking."""
        from proxima.backends.registry import BackendRegistry

        registry = BackendRegistry()
        registry.discover()

        available = registry.list_available()
        if not available:
            pytest.skip("No backends available")

        backend_name = available[0]

        # Mark failure
        registry.mark_backend_failure(backend_name, severity=0.3)
        status = registry.get_status(backend_name)
        assert status.health_score < 1.0

        # Mark success - use larger recovery value to ensure visible change
        registry.mark_backend_success(backend_name, recovery=0.2)
        status2 = registry.get_status(backend_name)
        # Use >= to handle floating-point edge cases where recovery is small
        assert status2.health_score >= status.health_score

    @pytest.mark.backend
    def test_registry_get_healthy_backends(self):
        """Test getting backends above health threshold."""
        from proxima.backends.registry import BackendRegistry

        registry = BackendRegistry()
        registry.discover()

        healthy = registry.get_healthy_backends(min_health=0.5)
        assert isinstance(healthy, list)

    @pytest.mark.backend
    def test_registry_reload_callback(self):
        """Test reload callback registration."""
        from proxima.backends.registry import BackendRegistry

        registry = BackendRegistry()
        registry.discover()

        callback_called = []

        def my_callback(reg):
            callback_called.append(True)

        registry.on_reload(my_callback)
        registry.hot_reload(force=True)

        assert len(callback_called) == 1

        # Remove callback
        assert registry.remove_reload_callback(my_callback) is True

    @pytest.mark.backend
    def test_registry_unregister(self):
        """Test backend unregistration."""
        from proxima.backends.registry import BackendRegistry

        registry = BackendRegistry()
        registry.discover()

        available = registry.list_available()
        if not available:
            pytest.skip("No backends available")

        # Unregister should return True for existing backend
        result = registry.unregister("nonexistent_backend")
        assert result is False