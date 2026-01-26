"""Backend Controller for Proxima TUI.

Handles backend registry integration, health checks, and selection.
"""

from typing import Optional, Callable, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from ..state import TUIState
from ..state.tui_state import BackendStatus
from ..state.events import BackendHealthChanged, BackendSelected

try:
    from proxima.backends.registry import BackendRegistry, HealthCheckResult
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    HealthCheckResult = None


# Supported backends
SUPPORTED_BACKENDS = [
    {
        "name": "lret",
        "display_name": "LRET",
        "description": "Local Realistic Entanglement Theory",
        "simulators": ["default"],
    },
    {
        "name": "cirq",
        "display_name": "Cirq",
        "description": "Google's quantum framework",
        "simulators": ["statevector", "density_matrix"],
    },
    {
        "name": "qiskit",
        "display_name": "Qiskit Aer",
        "description": "IBM quantum simulator",
        "simulators": ["statevector", "density_matrix"],
    },
    {
        "name": "cuquantum",
        "display_name": "cuQuantum",
        "description": "NVIDIA GPU acceleration",
        "simulators": ["statevector"],
        "requires_gpu": True,
    },
    {
        "name": "qsim",
        "display_name": "qsim",
        "description": "Google's high-performance simulator",
        "simulators": ["statevector"],
    },
    {
        "name": "quest",
        "display_name": "QuEST",
        "description": "Quantum Exact Simulation Toolkit",
        "simulators": ["statevector", "density_matrix"],
    },
]


class BackendController:
    """Controller for backend management.
    
    Handles listing backends, checking health, and selecting active backend.
    """
    
    def __init__(self, state: TUIState):
        """Initialize the backend controller.
        
        Args:
            state: The TUI state instance
        """
        self.state = state
        self._registry = None
        if CORE_AVAILABLE:
            try:
                self._registry = BackendRegistry()
                self._registry.discover()
            except Exception:
                pass  # Core not available, use simulated mode
        self._event_callbacks: List[Callable] = []
        
        # Initialize backend statuses
        self._initialize_backends()
    
    def _initialize_backends(self) -> None:
        """Initialize backend status entries."""
        for backend in SUPPORTED_BACKENDS:
            status = BackendStatus(
                name=backend["name"],
                status="unknown",
                available=False,
                simulator=backend["simulators"][0] if backend["simulators"] else None,
            )
            self.state.set_backend_status(backend["name"], status)
    
    @property
    def active_backend(self) -> Optional[BackendStatus]:
        """Get the active backend status."""
        if self.state.active_backend_name:
            return self.state.get_backend_status(self.state.active_backend_name)
        return None
    
    def get_all_backends(self) -> List[Dict[str, Any]]:
        """Get all backend definitions.
        
        Returns:
            List of backend definitions
        """
        return SUPPORTED_BACKENDS.copy()
    
    def get_backend_status(self, name: str) -> Optional[BackendStatus]:
        """Get status for a specific backend.
        
        Args:
            name: Backend name
        
        Returns:
            Backend status or None
        """
        return self.state.get_backend_status(name)
    
    def get_healthy_backends(self) -> List[str]:
        """Get list of healthy backend names.
        
        Returns:
            List of healthy backend names
        """
        return self.state.get_healthy_backends()
    
    def select_backend(self, name: str, simulator: Optional[str] = None) -> bool:
        """Select a backend as active.
        
        Args:
            name: Backend name
            simulator: Optional simulator type
        
        Returns:
            True if selection was successful
        """
        status = self.state.get_backend_status(name)
        if not status:
            return False
        
        # Update active backend
        self.state.active_backend_name = name
        self.state.current_backend = name
        
        if simulator:
            status.simulator = simulator
            self.state.current_simulator = simulator
        
        self._emit_event(BackendSelected(
            backend=name,
            simulator=simulator or status.simulator or "statevector",
        ))
        
        return True
    
    def check_health(self, name: Optional[str] = None) -> Dict[str, str]:
        """Check health of backends.
        
        Args:
            name: Specific backend name (checks all if None)
        
        Returns:
            Dict of backend names to health status
        """
        results = {}
        
        backends_to_check = [name] if name else [b["name"] for b in SUPPORTED_BACKENDS]
        
        for backend_name in backends_to_check:
            previous_status = self.state.get_backend_status(backend_name)
            previous_health = previous_status.status if previous_status else "unknown"
            
            # Check backend health via Proxima core if available
            if self._registry and CORE_AVAILABLE:
                try:
                    result = self._registry.check_backend_health(backend_name)
                    if result.success:
                        new_health = "healthy"
                    elif result.available:
                        new_health = "degraded"
                    else:
                        new_health = "unavailable"
                    response_time = result.response_time_ms if hasattr(result, 'response_time_ms') else 45.0
                except Exception:
                    new_health = self._simulate_health_check(backend_name)
                    response_time = 45.0
            else:
                # Fallback to simulated health check
                new_health = self._simulate_health_check(backend_name)
                response_time = 45.0

            # Update status
            if previous_status:
                previous_status.status = new_health
                previous_status.available = new_health in ["healthy", "degraded"]
                previous_status.response_time_ms = response_time
            
            results[backend_name] = new_health
            
            # Emit event if health changed
            if new_health != previous_health:
                self._emit_event(BackendHealthChanged(
                    backend=backend_name,
                    previous_status=previous_health,
                    current_status=new_health,
                    response_time_ms=response_time,
                ))
        
        return results
    
    def _simulate_health_check(self, name: str) -> str:
        """Simulate a health check result.
        
        Args:
            name: Backend name
        
        Returns:
            Simulated health status
        """
        # Simulate some backends being healthy
        always_healthy = ["lret", "cirq", "qiskit"]
        if name in always_healthy:
            return "healthy"
        
        # GPU backends need GPU
        gpu_backends = ["cuquantum"]
        if name in gpu_backends:
            return "unavailable"
        
        # Others are unknown
        return "unknown"
    
    def get_backend_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info for a backend.
        
        Args:
            name: Backend name
        
        Returns:
            Backend info dict or None
        """
        for backend in SUPPORTED_BACKENDS:
            if backend["name"] == name:
                status = self.state.get_backend_status(name)
                return {
                    **backend,
                    "status": status.status if status else "unknown",
                    "available": status.available if status else False,
                    "response_time_ms": status.response_time_ms if status else None,
                }
        return None
    
    def get_simulators_for_backend(self, name: str) -> List[str]:
        """Get available simulators for a backend.
        
        Args:
            name: Backend name
        
        Returns:
            List of simulator names
        """
        for backend in SUPPORTED_BACKENDS:
            if backend["name"] == name:
                return backend.get("simulators", [])
        return []
    
    def on_event(self, callback: Callable) -> None:
        """Register an event callback.
        
        Args:
            callback: Function to call on events
        """
        self._event_callbacks.append(callback)
    
    def _emit_event(self, event: Any) -> None:
        """Emit an event to callbacks.
        
        Args:
            event: Event to emit
        """
        for callback in self._event_callbacks:
            callback(event)
