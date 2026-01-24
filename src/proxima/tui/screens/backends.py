"""Backends screen for Proxima TUI.

Backend management and comparison.
"""

from textual.containers import Horizontal, Vertical, Container, Grid
from textual.widgets import Static, Button, DataTable
from rich.text import Text

from .base import BaseScreen
from ..styles.theme import get_theme
from ..styles.icons import get_health_icon


class BackendsScreen(BaseScreen):
    """Backend management screen.
    
    Shows:
    - List of backends with health status
    - Backend details
    - Performance comparison
    """
    
    SCREEN_NAME = "backends"
    SCREEN_TITLE = "Backend Management"
    
    DEFAULT_CSS = """
    BackendsScreen .backends-grid {
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
        padding: 1;
    }
    
    BackendsScreen .backend-card {
        height: 8;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface;
    }
    
    BackendsScreen .backend-card:hover {
        border: solid $primary;
    }
    
    BackendsScreen .backend-card.-selected {
        border: solid $primary;
        background: $surface-lighten-1;
    }
    
    BackendsScreen .backend-name {
        text-style: bold;
    }
    
    BackendsScreen .backend-status {
        margin-top: 1;
    }
    
    BackendsScreen .actions-section {
        height: auto;
        layout: horizontal;
        padding: 1;
        border-top: solid $primary-darken-3;
    }
    
    BackendsScreen .action-btn {
        margin-right: 1;
    }
    """
    
    def compose_main(self):
        """Compose the backends screen content."""
        with Vertical(classes="main-content"):
            # Title
            yield Static(
                "Available Backends",
                classes="section-title",
            )
            
            # Backend cards grid
            with Grid(classes="backends-grid"):
                yield BackendCard("lret", "LRET", "Local Realistic Entanglement Theory", "healthy")
                yield BackendCard("cirq", "Cirq", "Google's quantum framework", "healthy")
                yield BackendCard("qiskit", "Qiskit Aer", "IBM quantum simulator", "healthy")
                yield BackendCard("cuquantum", "cuQuantum", "NVIDIA GPU acceleration", "unavailable")
                yield BackendCard("qsim", "qsim", "High-performance simulator", "unknown")
                yield BackendCard("quest", "QuEST", "Quantum Exact Simulation Toolkit", "unknown")
            
            # Actions
            with Horizontal(classes="actions-section"):
                yield Button("Run Health Check", id="btn-health", classes="action-btn", variant="primary")
                yield Button("Compare Performance", id="btn-compare", classes="action-btn")
                yield Button("View Metrics", id="btn-metrics", classes="action-btn")
                yield Button("Configure", id="btn-configure", classes="action-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-health":
            self.notify("Running health checks...")
        elif button_id == "btn-compare":
            self.notify("Opening performance comparison...")
        elif button_id == "btn-metrics":
            self.notify("Loading metrics...")
        elif button_id == "btn-configure":
            self.notify("Opening backend configuration...")


class BackendCard(Static):
    """A card displaying backend information."""
    
    def __init__(
        self,
        backend_id: str,
        name: str,
        description: str,
        status: str,
        **kwargs,
    ):
        """Initialize the backend card."""
        super().__init__(**kwargs)
        self.backend_id = backend_id
        self.backend_name = name
        self.description = description
        self.status = status
        self.classes = "backend-card"
    
    def render(self) -> Text:
        """Render the backend card."""
        theme = get_theme()
        text = Text()
        
        # Status icon and name
        icon = get_health_icon(self.status)
        color = theme.get_health_color(self.status)
        
        text.append(icon, style=f"bold {color}")
        text.append(" ")
        text.append(self.backend_name, style=f"bold {theme.fg_base}")
        text.append("\n")
        
        # Description
        text.append(self.description, style=theme.fg_muted)
        text.append("\n\n")
        
        # Status text
        status_text = self.status.capitalize()
        if self.status == "healthy":
            text.append(f"● {status_text}", style=f"bold {color}")
        elif self.status == "unavailable":
            text.append(f"○ Not Available", style=color)
        else:
            text.append(f"○ {status_text}", style=color)
        
        return text
