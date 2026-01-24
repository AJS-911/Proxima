"""Dashboard screen for Proxima TUI.

Main landing screen with overview and quick actions.
"""

from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Button, DataTable
from rich.text import Text
from rich.panel import Panel

from .base import BaseScreen
from ..styles.theme import get_theme
from ..components.logo import Logo


class DashboardScreen(BaseScreen):
    """Main dashboard screen.
    
    Shows:
    - Welcome message
    - Quick actions
    - Recent sessions
    - System health
    """
    
    SCREEN_NAME = "dashboard"
    SCREEN_TITLE = "Dashboard"
    
    DEFAULT_CSS = """
    DashboardScreen .welcome-section {
        height: auto;
        margin-bottom: 2;
    }
    
    DashboardScreen .welcome-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    DashboardScreen .welcome-subtitle {
        color: $text-muted;
    }
    
    DashboardScreen .quick-actions {
        height: auto;
        margin-bottom: 2;
        padding: 1;
        border: solid $primary;
        background: $surface;
    }
    
    DashboardScreen .quick-actions-title {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    DashboardScreen .action-buttons {
        layout: horizontal;
        height: auto;
    }
    
    DashboardScreen .action-button {
        margin-right: 1;
        min-width: 18;
    }
    
    DashboardScreen .sessions-section {
        height: 1fr;
        margin-bottom: 1;
        border: solid $primary-darken-2;
        background: $surface;
    }
    
    DashboardScreen .sessions-title {
        color: $text-muted;
        padding: 1;
        border-bottom: solid $primary-darken-3;
    }
    
    DashboardScreen .health-section {
        height: auto;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface;
    }
    """
    
    def compose_main(self):
        """Compose the dashboard content."""
        theme = get_theme()
        
        with Vertical(classes="main-content"):
            # Welcome section
            with Vertical(classes="welcome-section"):
                yield Static(
                    "Welcome to Proxima",
                    classes="welcome-title",
                )
                yield Static(
                    "Intelligent Quantum Simulation Orchestration",
                    classes="welcome-subtitle",
                )
            
            # Quick Actions
            with Container(classes="quick-actions"):
                yield Static("Quick Actions", classes="quick-actions-title")
                with Horizontal(classes="action-buttons"):
                    yield Button(
                        "[1] Run Simulation",
                        id="btn-run",
                        classes="action-button",
                        variant="primary",
                    )
                    yield Button(
                        "[2] Compare Backends",
                        id="btn-compare",
                        classes="action-button",
                    )
                    yield Button(
                        "[3] View Results",
                        id="btn-results",
                        classes="action-button",
                    )
                    yield Button(
                        "[4] Manage Sessions",
                        id="btn-sessions",
                        classes="action-button",
                    )
                    yield Button(
                        "[5] Configure",
                        id="btn-config",
                        classes="action-button",
                    )
                    yield Button(
                        "[?] Help",
                        id="btn-help",
                        classes="action-button",
                    )
            
            # Recent Sessions
            with Vertical(classes="sessions-section"):
                yield Static("Recent Sessions", classes="sessions-title")
                yield RecentSessionsTable()
            
            # System Health
            yield SystemHealthBar(classes="health-section")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-run":
            self.app.action_open_commands()
        elif button_id == "btn-compare":
            self.app.action_goto_backends()
        elif button_id == "btn-results":
            self.app.action_goto_results()
        elif button_id == "btn-sessions":
            # Open session dialog
            pass
        elif button_id == "btn-config":
            self.app.action_goto_settings()
        elif button_id == "btn-help":
            self.app.action_show_help()


class RecentSessionsTable(DataTable):
    """Table displaying recent sessions."""
    
    DEFAULT_CSS = """
    RecentSessionsTable {
        height: 1fr;
        margin: 1;
    }
    """
    
    def on_mount(self) -> None:
        """Set up the table."""
        self.add_columns("ID", "Task", "Backend", "Status", "Time")
        self.cursor_type = "row"
        
        # Add sample data
        self._populate_sample_data()
    
    def _populate_sample_data(self) -> None:
        """Populate with sample session data."""
        theme = get_theme()
        
        sample_sessions = [
            ("a1b2c3d4", "Bell State", "Cirq", "✓ Done", "12s ago"),
            ("e5f6g7h8", "GHZ 4-qubit", "Qiskit", "✓ Done", "2m ago"),
            ("i9j0k1l2", "Comparison Run", "Multi", "✓ Done", "15m ago"),
            ("m3n4o5p6", "Grover Search", "Cirq", "✓ Done", "1h ago"),
            ("q7r8s9t0", "VQE Optimization", "Qiskit", "⏸ Paused", "2h ago"),
        ]
        
        for session in sample_sessions:
            self.add_row(*session)


class SystemHealthBar(Static):
    """System health overview bar."""
    
    def render(self) -> Text:
        """Render the health bar."""
        theme = get_theme()
        text = Text()
        
        # CPU
        text.append("CPU: ", style=theme.fg_muted)
        text.append("23%", style=f"bold {theme.success}")
        text.append("  │  ", style=theme.border)
        
        # Memory
        text.append("Memory: ", style=theme.fg_muted)
        text.append("52%", style=f"bold {theme.success}")
        text.append("  │  ", style=theme.border)
        
        # Backends
        text.append("Backends: ", style=theme.fg_muted)
        text.append("3/6 healthy", style=f"bold {theme.success}")
        
        return text
