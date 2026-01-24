"""Backward compatibility shim for proxima.tui.widgets.

This module provides backward compatibility with the old TUI architecture.
New code should use proxima.tui.components instead.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time

from textual.widget import Widget
from textual.widgets import Static
from textual.screen import ModalScreen


# ======================== ENUMS ========================


class StatusLevel(Enum):
    """Status level enum for widgets."""
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class BackendStatus(Enum):
    """Backend status enum."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    UNKNOWN = "unknown"


# ======================== DATACLASSES ========================


@dataclass
class StatusItem:
    """Status item for display."""
    label: str
    value: str
    level: StatusLevel = StatusLevel.INFO
    icon: str = ""


@dataclass
class LogEntry:
    """Log entry for log viewers."""
    timestamp: float
    level: str
    message: str
    component: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def format_level(self) -> str:
        """Format level as uppercase string."""
        return self.level.upper()
    
    def format_timestamp(self) -> str:
        """Format timestamp as string."""
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))


@dataclass
class BackendInfo:
    """Backend information."""
    name: str
    backend_type: str
    status: BackendStatus = BackendStatus.UNKNOWN
    total_executions: int = 0
    avg_latency_ms: Optional[float] = None
    last_error: str = ""
    last_used: Optional[str] = None


# ======================== WIDGETS ========================


class StatusPanel(Static):
    """Status panel widget for backward compatibility."""
    
    def __init__(
        self,
        title: str = "Status",
        items: Optional[List[StatusItem]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._title = title
        self._items = items or []


class LogPanel(Static):
    """Log panel widget."""
    
    def __init__(
        self,
        title: str = "Logs",
        max_lines: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._title = title
        self._max_lines = max_lines
        self._logs: List[str] = []


class LogViewer(Static):
    """Log viewer widget with scrolling support."""
    
    def __init__(
        self,
        max_entries: int = 1000,
        auto_scroll: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._max_entries = max_entries
        self._auto_scroll = auto_scroll
        self._entries: List[LogEntry] = []


class BackendCard(Static):
    """Backend status card."""
    
    def __init__(
        self,
        backend: Optional[BackendInfo] = None,
        backend_name: str = "",
        status: str = "unknown",
        details: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._backend = backend
        self._name = backend_name or (backend.name if backend else "")
        self._status = status
        self._details = details or {}


class ResultsTable(Static):
    """Results table widget."""
    
    def __init__(
        self,
        title: str = "Results",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._title = title
        self._results: List[Dict[str, Any]] = []


class ExecutionTimer(Static):
    """Execution timer widget."""
    
    def __init__(
        self,
        label: str = "Timer",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._label = label
        self._start_time: Optional[float] = None
        self._elapsed = 0.0
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time."""
        return self._elapsed


class MetricDisplay(Static):
    """Single metric display widget."""
    
    def __init__(
        self,
        label: str = "Metric",
        value: str = "",
        unit: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._label = label
        self._value = value
        self._unit = unit
    
    @property
    def value(self) -> str:
        """Get the value."""
        return self._value


class MetricsDisplay(Static):
    """Multiple metrics display widget."""
    
    def __init__(
        self,
        title: str = "Metrics",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._title = title
        self._metrics: Dict[str, Any] = {}


class ExecutionProgress(Static):
    """Execution progress widget."""
    
    def __init__(
        self,
        title: str = "Progress",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._title = title
        self._progress = 0.0
    
    @property
    def progress(self) -> float:
        """Get progress value."""
        return self._progress


class StatusIndicator(Static):
    """Status indicator with icon."""
    
    ICONS = {
        "success": "✓",
        "error": "✗",
        "warning": "⚠",
        "info": "ℹ",
        "pending": "○",
        "running": "●",
    }
    
    def __init__(
        self,
        status: str = "info",
        label: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._status = status
        self._label = label
    
    @property
    def status(self) -> str:
        """Get status."""
        return self._status


class HelpModal(ModalScreen):
    """Help modal widget."""
    
    def __init__(self, **kwargs):
        super().__init__()


class ConfigInput(Static):
    """Configuration input widget."""
    
    def __init__(
        self,
        key: str,
        label: str,
        value: str = "",
        input_type: str = "text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._key = key
        self._label = label
        self._value = value
        self._input_type = input_type


class ConfigToggle(Static):
    """Configuration toggle widget."""
    
    def __init__(
        self,
        key: str,
        label: str,
        value: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._key = key
        self._label = label
        self._value = value


class ExecutionCard(Static):
    """Execution history card."""
    
    def __init__(
        self,
        execution_id: str,
        backend: str,
        status: str,
        duration_ms: float = 0.0,
        timestamp: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._id = execution_id
        self._backend = backend
        self._status = status
        self._duration_ms = duration_ms
        self._timestamp = timestamp


class ProgressBar(Static):
    """Progress bar widget."""
    
    def __init__(
        self,
        label: str = "Progress",
        total: float = 100.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._label = label
        self._total = total
        self._current = 0.0


__all__ = [
    # Enums
    "StatusLevel",
    "BackendStatus",
    # Dataclasses
    "StatusItem",
    "LogEntry",
    "BackendInfo",
    # Widgets
    "StatusPanel",
    "LogPanel",
    "LogViewer",
    "BackendCard",
    "ResultsTable",
    "ExecutionTimer",
    "MetricDisplay",
    "MetricsDisplay",
    "ExecutionProgress",
    "StatusIndicator",
    "HelpModal",
    "ConfigInput",
    "ConfigToggle",
    "ExecutionCard",
    "ProgressBar",
]
