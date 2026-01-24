"""Backend selection dialog for Proxima TUI.

Dialog for selecting and configuring simulation backends.
"""

from dataclasses import dataclass
from typing import List, Optional

from textual.screen import ModalScreen
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Input, ListView, ListItem, Label, Select
from textual import on
from rich.text import Text

from ...styles.theme import get_theme
from ...styles.icons import ICON_BACKEND, get_health_icon


@dataclass
class BackendInfo:
    """Information about a simulation backend."""
    
    name: str
    description: str
    simulators: List[str]
    health: str = "unknown"  # healthy, degraded, unhealthy, unknown
    response_time_ms: Optional[float] = None


# Default available backends
DEFAULT_BACKENDS = [
    BackendInfo("lret", "Local Realistic Entanglement Theory", ["default"], "healthy", 38),
    BackendInfo("cirq", "Google's quantum framework", ["state_vector", "density_matrix"], "healthy", 45),
    BackendInfo("qiskit", "IBM quantum simulator", ["aer", "state_vector", "density_matrix"], "healthy", 52),
    BackendInfo("cuquantum", "NVIDIA GPU acceleration", ["custatevec", "cutensornet"], "unavailable"),
    BackendInfo("qsim", "Google's high-performance simulator", ["default"], "unknown"),
    BackendInfo("quest", "Quantum Exact Simulation Toolkit", ["default"], "unknown"),
]


class BackendsDialog(ModalScreen):
    """Dialog for selecting simulation backends.
    
    Features:
    - List available backends with health status
    - Select simulator for backend
    - Show response times
    """
    
    DEFAULT_CSS = """
    BackendsDialog {
        align: center middle;
    }
    
    BackendsDialog > .dialog-container {
        width: 70;
        height: 28;
        border: thick $primary;
        background: $surface;
    }
    
    BackendsDialog .dialog-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        padding: 1;
        border-bottom: solid $primary-darken-3;
    }
    
    BackendsDialog .backends-list {
        height: 1fr;
        margin: 1;
    }
    
    BackendsDialog .simulator-section {
        height: auto;
        padding: 1;
        border-top: solid $primary-darken-3;
    }
    
    BackendsDialog .footer {
        height: auto;
        padding: 1;
        border-top: solid $primary-darken-3;
        text-align: center;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "select", "Select"),
        ("up", "move_up", "Up"),
        ("down", "move_down", "Down"),
        ("h", "health_check", "Health Check"),
    ]
    
    def __init__(
        self,
        backends: Optional[List[BackendInfo]] = None,
        current_backend: Optional[str] = None,
        current_simulator: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the backends dialog.
        
        Args:
            backends: Available backends (uses defaults if None)
            current_backend: Currently selected backend
            current_simulator: Currently selected simulator
        """
        super().__init__(**kwargs)
        self.backends = backends or DEFAULT_BACKENDS
        self.current_backend = current_backend
        self.current_simulator = current_simulator
        self.selected_index = 0
    
    def compose(self):
        """Compose the dialog layout."""
        with Vertical(classes="dialog-container"):
            yield Static(f"{ICON_BACKEND} Select Backend", classes="dialog-title")
            yield BackendsListView(
                self.backends,
                self.current_backend,
                classes="backends-list",
            )
            
            # Simulator selection
            with Horizontal(classes="simulator-section"):
                yield Static("Simulator: ", classes="option-label")
                yield Select(
                    [(s, s) for s in self._get_current_simulators()],
                    value=self.current_simulator or "default",
                    id="simulator-select",
                )
            
            yield BackendsFooter(classes="footer")
    
    def _get_current_simulators(self) -> List[str]:
        """Get simulators for currently selected backend."""
        if self.selected_index < len(self.backends):
            return self.backends[self.selected_index].simulators
        return ["default"]
    
    def _update_simulator_options(self) -> None:
        """Update simulator select options."""
        simulators = self._get_current_simulators()
        select = self.query_one("#simulator-select", Select)
        select.set_options([(s, s) for s in simulators])
        if simulators:
            select.value = simulators[0]
    
    def action_cancel(self) -> None:
        """Cancel and close."""
        self.dismiss(None)
    
    def action_select(self) -> None:
        """Select current backend."""
        if self.selected_index < len(self.backends):
            backend = self.backends[self.selected_index]
            simulator = self.query_one("#simulator-select", Select).value
            self.dismiss({"backend": backend, "simulator": simulator})
    
    def action_move_up(self) -> None:
        """Move selection up."""
        if self.selected_index > 0:
            self.selected_index -= 1
            self.query_one(BackendsListView).index = self.selected_index
            self._update_simulator_options()
    
    def action_move_down(self) -> None:
        """Move selection down."""
        if self.selected_index < len(self.backends) - 1:
            self.selected_index += 1
            self.query_one(BackendsListView).index = self.selected_index
            self._update_simulator_options()
    
    def action_health_check(self) -> None:
        """Trigger health check."""
        self.notify("Running health checks...")


class BackendsListView(ListView):
    """List view for backends."""
    
    def __init__(
        self,
        backends: List[BackendInfo],
        current: Optional[str],
        **kwargs,
    ):
        """Initialize the list."""
        super().__init__(**kwargs)
        self._backends = backends
        self._current = current
    
    def on_mount(self) -> None:
        """Populate the list."""
        for backend in self._backends:
            is_current = backend.name == self._current
            self.append(BackendItem(backend, is_current))


class BackendItem(ListItem):
    """A single backend item in the list."""
    
    DEFAULT_CSS = """
    BackendItem {
        height: 3;
        padding: 0 1;
    }
    
    BackendItem:hover {
        background: $primary-darken-3;
    }
    """
    
    def __init__(self, backend: BackendInfo, is_current: bool = False, **kwargs):
        """Initialize the backend item."""
        super().__init__(**kwargs)
        self.backend = backend
        self.is_current = is_current
    
    def compose(self):
        """Compose the item content."""
        yield Label(self._render_content())
    
    def _render_content(self) -> Text:
        """Render the backend item content."""
        theme = get_theme()
        text = Text()
        
        # Health icon
        icon = get_health_icon(self.backend.health)
        color = theme.get_health_color(self.backend.health)
        text.append(icon, style=f"bold {color}")
        text.append(" ")
        
        # Backend name
        text.append(self.backend.name, style=f"bold {theme.fg_base}")
        
        # Current indicator
        if self.is_current:
            text.append(" (active)", style=theme.accent)
        
        # Response time
        if self.backend.response_time_ms:
            text.append(f"  {self.backend.response_time_ms:.0f}ms", style=theme.fg_subtle)
        
        text.append("\n  ")
        
        # Description
        text.append(self.backend.description, style=theme.fg_muted)
        
        return text


class BackendsFooter(Static):
    """Footer with keybindings."""
    
    def render(self) -> Text:
        """Render the footer."""
        theme = get_theme()
        text = Text()
        
        bindings = [
            ("↑↓", "navigate"),
            ("enter", "select"),
            ("h", "health check"),
            ("esc", "cancel"),
        ]
        
        for i, (key, desc) in enumerate(bindings):
            if i > 0:
                text.append(" │ ", style=theme.border)
            text.append(key, style=f"bold {theme.accent}")
            text.append(f" {desc}", style=theme.fg_muted)
        
        return text
