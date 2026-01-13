"""Backend registry for managing available backends.

Implements Step 2.1 Backend Registry with:
- Dynamic backend discovery
- Capability caching
- Health status tracking
- Hot-reload support for refreshing backends without restart
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from proxima.backends.base import BaseBackendAdapter, Capabilities
from proxima.backends.cirq_adapter import CirqBackendAdapter
from proxima.backends.cuquantum_adapter import CuQuantumAdapter
from proxima.backends.lret import LRETBackendAdapter
from proxima.backends.qiskit_adapter import QiskitBackendAdapter
from proxima.backends.qsim_adapter import QsimAdapter  # Step 3.5: qsim added
from proxima.backends.quest_adapter import QuestBackendAdapter

logger = logging.getLogger(__name__)

# Type alias for reload callbacks
ReloadCallback = Callable[["BackendRegistry"], None]


@dataclass
class BackendStatus:
    name: str
    available: bool
    adapter: BaseBackendAdapter | None = None
    capabilities: Capabilities | None = None
    version: str | None = None
    reason: str | None = None
    last_checked: float = field(default_factory=time.time)
    health_score: float = 1.0  # 0.0 to 1.0, tracks reliability


class BackendRegistry:
    """Maintains discovery and lookup of backend adapters.
    
    Supports hot-reload for refreshing backend discovery without restart.
    Thread-safe for concurrent access during reload operations.
    """

    # How long before a backend status is considered stale (seconds)
    CACHE_TTL_SECONDS: float = 300.0  # 5 minutes

    def __init__(self) -> None:
        self._statuses: dict[str, BackendStatus] = {}
        self._lock = threading.RLock()
        self._reload_callbacks: list[ReloadCallback] = []
        self._last_discovery: float = 0.0
        self._discovery_count: int = 0

    def register(self, adapter: BaseBackendAdapter) -> None:
        name = adapter.get_name()
        capabilities = adapter.get_capabilities()
        version = self._safe_get_version(adapter)
        with self._lock:
            self._statuses[name] = BackendStatus(
                name=name,
                available=True,
                adapter=adapter,
                capabilities=capabilities,
                version=version,
                reason=None,
                last_checked=time.time(),
                health_score=1.0,
            )
            
    def unregister(self, name: str) -> bool:
        """Unregister a backend by name. Returns True if removed."""
        with self._lock:
            if name in self._statuses:
                del self._statuses[name]
                return True
            return False
    
    def on_reload(self, callback: ReloadCallback) -> None:
        """Register callback to be notified after hot-reload."""
        self._reload_callbacks.append(callback)
        
    def remove_reload_callback(self, callback: ReloadCallback) -> bool:
        """Remove a reload callback. Returns True if removed."""
        try:
            self._reload_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def discover(self) -> None:
        """Discover known backends, cache capabilities, and mark health status."""
        with self._lock:
            self._discover_internal()
            
    def _discover_internal(self) -> None:
        """Internal discovery implementation (must hold lock)."""
        self._statuses = {}
        candidates: list[type[BaseBackendAdapter]] = [
            LRETBackendAdapter,
            CirqBackendAdapter,
            QiskitBackendAdapter,
            QuestBackendAdapter,
            CuQuantumAdapter,  # Step 2.3: cuQuantum added to registry
            QsimAdapter,  # Step 3.5: qsim added to registry
        ]

        for adapter_cls in candidates:
            status = self._init_adapter(adapter_cls)
            self._statuses[status.name] = status
            
        self._last_discovery = time.time()
        self._discovery_count += 1
        
    def hot_reload(self, force: bool = False) -> dict[str, str]:
        """Hot-reload backend discovery without full restart.
        
        This method:
        1. Refreshes module imports for backend adapters
        2. Re-runs discovery to detect newly available backends
        3. Preserves health scores for backends that remain available
        4. Notifies registered callbacks after reload
        
        Args:
            force: If True, reload even if cache is fresh
            
        Returns:
            Dict mapping backend names to status changes ("added", "removed", "changed", "unchanged")
        """
        with self._lock:
            # Check if reload is needed
            elapsed = time.time() - self._last_discovery
            if not force and elapsed < self.CACHE_TTL_SECONDS:
                logger.debug(f"Skipping hot-reload, cache fresh ({elapsed:.1f}s < {self.CACHE_TTL_SECONDS}s)")
                return {}
                
            # Save previous state
            old_statuses = dict(self._statuses)
            old_available = {name for name, s in old_statuses.items() if s.available}
            
            # Refresh module imports for adapters
            adapter_modules = [
                "proxima.backends.lret",
                "proxima.backends.cirq_adapter",
                "proxima.backends.qiskit_adapter",
                "proxima.backends.quest_adapter",
                "proxima.backends.cuquantum_adapter",
                "proxima.backends.qsim_adapter",
            ]
            for mod_name in adapter_modules:
                try:
                    if mod_name in importlib.sys.modules:
                        importlib.reload(importlib.sys.modules[mod_name])
                except Exception as e:
                    logger.warning(f"Failed to reload module {mod_name}: {e}")
            
            # Re-run discovery
            self._discover_internal()
            
            # Calculate changes
            new_available = {name for name, s in self._statuses.items() if s.available}
            changes: dict[str, str] = {}
            
            all_names = old_available | new_available | set(old_statuses.keys()) | set(self._statuses.keys())
            for name in all_names:
                old_status = old_statuses.get(name)
                new_status = self._statuses.get(name)
                
                if old_status is None and new_status is not None:
                    changes[name] = "added"
                elif old_status is not None and new_status is None:
                    changes[name] = "removed"
                elif old_status and new_status:
                    if old_status.available != new_status.available:
                        changes[name] = "changed"
                        # Preserve health score if backend returns
                        if new_status.available and old_status.health_score < 1.0:
                            new_status.health_score = old_status.health_score
                    elif old_status.version != new_status.version:
                        changes[name] = "changed"
                    else:
                        changes[name] = "unchanged"
                        # Preserve health score
                        if old_status.health_score < 1.0:
                            new_status.health_score = old_status.health_score
            
            logger.info(
                f"Hot-reload complete: {len(changes)} backends processed, "
                f"{sum(1 for c in changes.values() if c != 'unchanged')} changed"
            )
        
        # Notify callbacks outside lock
        for callback in self._reload_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Reload callback error: {e}")
                
        return changes
        
    def refresh_backend(self, name: str) -> BackendStatus | None:
        """Refresh a single backend's status without full reload.
        
        Useful for checking if a specific backend became available.
        
        Args:
            name: Backend name to refresh
            
        Returns:
            Updated BackendStatus, or None if backend not found
        """
        adapter_map = {
            "lret": LRETBackendAdapter,
            "cirq": CirqBackendAdapter,
            "qiskit": QiskitBackendAdapter,
            "quest": QuestBackendAdapter,
            "cuquantum": CuQuantumAdapter,
            "qsim": QsimAdapter,
        }
        
        adapter_cls = adapter_map.get(name.lower())
        if not adapter_cls:
            return None
            
        with self._lock:
            old_status = self._statuses.get(name)
            new_status = self._init_adapter(adapter_cls)
            
            # Preserve health score if refreshing
            if old_status and old_status.health_score < 1.0:
                new_status.health_score = old_status.health_score
                
            self._statuses[new_status.name] = new_status
            return new_status
            
    def mark_backend_failure(self, name: str, severity: float = 0.1) -> None:
        """Record a backend failure to adjust health score.
        
        Args:
            name: Backend name
            severity: How much to reduce health (0.0 to 1.0)
        """
        with self._lock:
            status = self._statuses.get(name)
            if status:
                status.health_score = max(0.0, status.health_score - severity)
                logger.debug(f"Backend {name} health reduced to {status.health_score:.2f}")
                
    def mark_backend_success(self, name: str, recovery: float = 0.05) -> None:
        """Record a backend success to recover health score.
        
        Args:
            name: Backend name
            recovery: How much to increase health (0.0 to 1.0)
        """
        with self._lock:
            status = self._statuses.get(name)
            if status:
                status.health_score = min(1.0, status.health_score + recovery)
                
    def get_discovery_stats(self) -> dict:
        """Get statistics about backend discovery."""
        with self._lock:
            return {
                "discovery_count": self._discovery_count,
                "last_discovery": self._last_discovery,
                "cache_age_seconds": time.time() - self._last_discovery if self._last_discovery else None,
                "total_backends": len(self._statuses),
                "available_backends": len([s for s in self._statuses.values() if s.available]),
            }

    def _init_adapter(self, adapter_cls: type[BaseBackendAdapter]) -> BackendStatus:
        name = getattr(adapter_cls, "__name__", "unknown").lower()

        try:
            adapter = adapter_cls()
            name = adapter.get_name()
        except Exception as exc:  # pragma: no cover - defensive
            return BackendStatus(
                name=name,
                available=False,
                adapter=None,
                capabilities=None,
                version=None,
                reason=f"initialization failed: {exc}",
            )

        try:
            if not adapter.is_available():
                return BackendStatus(
                    name=name,
                    available=False,
                    adapter=None,
                    capabilities=None,
                    version=None,
                    reason=self._dependency_reason(adapter_cls),
                )
        except Exception as exc:  # pragma: no cover - defensive
            return BackendStatus(
                name=name,
                available=False,
                adapter=None,
                capabilities=None,
                version=None,
                reason=f"availability check failed: {exc}",
            )

        try:
            capabilities = adapter.get_capabilities()
        except Exception as exc:  # pragma: no cover - defensive
            return BackendStatus(
                name=name,
                available=False,
                adapter=None,
                capabilities=None,
                version=None,
                reason=f"capabilities check failed: {exc}",
            )

        version = self._safe_get_version(adapter)
        return BackendStatus(
            name=name,
            available=True,
            adapter=adapter,
            capabilities=capabilities,
            version=version,
            reason=None,
        )

    def _dependency_reason(self, adapter_cls: type[BaseBackendAdapter]) -> str:
        dependency_map = {
            "lretbackendadapter": ["lret"],
            "cirqbackendadapter": ["cirq"],
            "qiskitbackendadapter": ["qiskit", "qiskit_aer"],
            "questbackendadapter": ["pyQuEST"],
            "cuquantumadapter": [
                "qiskit",
                "qiskit_aer",
                "cuquantum",
            ],  # cuQuantum dependencies
            "qsimadapter": ["cirq", "qsimcirq"],  # Step 3.5: qsim dependencies
        }
        missing = []
        for dep in dependency_map.get(adapter_cls.__name__.lower(), []):
            if importlib.util.find_spec(dep) is None:
                missing.append(dep)
        if missing:
            return f"missing dependency: {', '.join(missing)}"
        return "adapter reported unavailable"

    def _safe_get_version(self, adapter: BaseBackendAdapter) -> str:
        try:
            return adapter.get_version()
        except Exception as exc:  # pragma: no cover - defensive
            return f"unknown (version check failed: {exc})"

    def get(self, name: str) -> BaseBackendAdapter:
        with self._lock:
            status = self._statuses.get(name)
            if not status:
                raise KeyError(f"Backend '{name}' not registered")
            if not status.available or not status.adapter:
                raise KeyError(
                    f"Backend '{name}' is unavailable: {status.reason or 'unknown reason'}"
                )
            return status.adapter

    def is_available(self, name: str) -> bool:
        with self._lock:
            status = self._statuses.get(name)
            return bool(status and status.available)

    def list_available(self) -> list[str]:
        with self._lock:
            return [name for name, status in self._statuses.items() if status.available]

    def list_statuses(self) -> list[BackendStatus]:
        with self._lock:
            return list(self._statuses.values())

    def get_capabilities(self, name: str) -> Capabilities:
        return self.get(name).get_capabilities()

    def get_status(self, name: str) -> BackendStatus:
        with self._lock:
            status = self._statuses.get(name)
            if not status:
                raise KeyError(f"Backend '{name}' not registered")
            return status
    
    def get_healthy_backends(self, min_health: float = 0.5) -> list[str]:
        """Get backends with health score above threshold."""
        with self._lock:
            return [
                name for name, status in self._statuses.items()
                if status.available and status.health_score >= min_health
            ]

    # ==========================================================================
    # Step 2.3: GPU-aware backend selection helpers
    # ==========================================================================

    def get_gpu_backends(self) -> list[str]:
        """Return list of GPU-enabled backends."""
        gpu_backends = []
        for name, status in self._statuses.items():
            if status.available and status.capabilities:
                if status.capabilities.supports_gpu:
                    gpu_backends.append(name)
        return gpu_backends

    def get_best_backend_for_circuit(
        self,
        qubit_count: int,
        simulation_type: str = "state_vector",
        prefer_gpu: bool = True,
    ) -> str | None:
        """Get best available backend for given circuit requirements.

        Args:
            qubit_count: Number of qubits in circuit
            simulation_type: "state_vector" or "density_matrix"
            prefer_gpu: Whether to prefer GPU backends

        Returns:
            Name of best backend, or None if none suitable
        """
        # Priority order based on simulation type and GPU preference
        # Step 3.5: qsim included in priority lists
        if simulation_type == "state_vector":
            if prefer_gpu:
                priority = ["cuquantum", "quest", "qsim", "qiskit", "cirq"]
            else:
                priority = ["qsim", "quest", "qiskit", "cirq"]
        elif simulation_type == "density_matrix":
            priority = ["quest", "cirq", "qiskit", "lret"]
        else:
            priority = ["qsim", "qiskit", "cirq", "quest"]

        for backend_name in priority:
            status = self._statuses.get(backend_name)
            if status and status.available and status.capabilities:
                if qubit_count <= status.capabilities.max_qubits:
                    return backend_name

        return None


backend_registry = BackendRegistry()
backend_registry.discover()
