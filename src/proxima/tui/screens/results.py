"""Results screen for Proxima TUI.

Results browser with probability visualization.
"""

from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Button, DataTable, ListView, ListItem, Label
from rich.text import Text

from .base import BaseScreen
from ..styles.theme import get_theme
from ..styles.icons import PROGRESS_FILLED, PROGRESS_EMPTY


class ResultsScreen(BaseScreen):
    """Results browser screen.
    
    Shows:
    - List of results
    - Result details with probability distribution
    - Export options
    """
    
    SCREEN_NAME = "results"
    SCREEN_TITLE = "Results Browser"
    
    DEFAULT_CSS = """
    ResultsScreen .results-list {
        width: 30;
        height: 100%;
        border-right: solid $primary-darken-2;
        background: $surface;
    }
    
    ResultsScreen .results-list-title {
        padding: 1;
        text-style: bold;
        border-bottom: solid $primary-darken-3;
    }
    
    ResultsScreen .result-detail {
        width: 1fr;
        height: 100%;
        padding: 1;
    }
    
    ResultsScreen .result-header {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid $primary;
        background: $surface;
    }
    
    ResultsScreen .probability-section {
        height: 1fr;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface-darken-1;
        overflow-y: auto;
    }
    
    ResultsScreen .actions-section {
        height: auto;
        layout: horizontal;
        margin-top: 1;
    }
    
    ResultsScreen .action-btn {
        margin-right: 1;
    }
    """
    
    def compose_main(self):
        """Compose the results screen content."""
        with Horizontal(classes="main-content"):
            # Results list
            with Vertical(classes="results-list"):
                yield Static("Results", classes="results-list-title")
                yield ResultsListView()
            
            # Result detail
            with Vertical(classes="result-detail"):
                yield ResultHeaderPanel(classes="result-header")
                yield ProbabilityDistribution(classes="probability-section")
                
                with Horizontal(classes="actions-section"):
                    yield Button("View Full Stats", id="btn-stats", classes="action-btn")
                    yield Button("Export JSON", id="btn-json", classes="action-btn")
                    yield Button("Export HTML", id="btn-html", classes="action-btn")
                    yield Button("Compare", id="btn-compare", classes="action-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-stats":
            self.notify("Opening full statistics...")
        elif button_id == "btn-json":
            self.notify("Exporting to JSON...")
        elif button_id == "btn-html":
            self.notify("Exporting to HTML...")
        elif button_id == "btn-compare":
            self.notify("Opening comparison view...")


class ResultsListView(ListView):
    """List of available results."""
    
    def on_mount(self) -> None:
        """Populate the results list."""
        # Sample results
        results = [
            "result_001.json",
            "result_002.json",
            "comparison_001.json",
            "bell_state_run.json",
            "ghz_4qubit.json",
        ]
        
        for result in results:
            self.append(ListItem(Label(result)))


class ResultHeaderPanel(Static):
    """Header panel showing result metadata."""
    
    def render(self) -> Text:
        """Render the result header."""
        theme = get_theme()
        text = Text()
        
        # Title line
        text.append("Simulation Results", style=f"bold {theme.primary}")
        text.append("\n\n")
        
        # Metadata
        text.append("Backend: ", style=theme.fg_muted)
        text.append("Cirq (StateVector)", style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        text.append("Qubits: ", style=theme.fg_muted)
        text.append("4", style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        text.append("Shots: ", style=theme.fg_muted)
        text.append("1024", style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        text.append("Time: ", style=theme.fg_muted)
        text.append("245ms", style=theme.fg_base)
        
        return text


class ProbabilityDistribution(Static):
    """Probability distribution visualization."""
    
    BAR_WIDTH = 40
    
    def render(self) -> Text:
        """Render the probability distribution."""
        theme = get_theme()
        text = Text()
        
        text.append("Probability Distribution:", style=f"bold {theme.fg_base}")
        text.append("\n\n")
        
        # Sample probability data
        probabilities = [
            ("|0000⟩", 48.2),
            ("|1111⟩", 47.1),
            ("|0011⟩", 2.4),
            ("|1100⟩", 1.8),
            ("|others⟩", 0.5),
        ]
        
        for state, prob in probabilities:
            # State label
            text.append(f"{state:<10}", style=f"bold {theme.accent}")
            
            # Bar
            filled = int(self.BAR_WIDTH * prob / 100)
            empty = self.BAR_WIDTH - filled
            
            text.append(PROGRESS_FILLED * filled, style=f"bold {theme.primary}")
            text.append(PROGRESS_EMPTY * empty, style=theme.fg_subtle)
            
            # Percentage
            text.append(f" {prob:>5.1f}%", style=theme.fg_muted)
            text.append("\n")
        
        # Pattern detection
        text.append("\n")
        text.append("Pattern: ", style=theme.fg_muted)
        text.append("GHZ State", style=f"bold {theme.success}")
        text.append(" (confidence: 96%)", style=theme.fg_muted)
        text.append("\n")
        
        # Statistics
        text.append("Entropy: ", style=theme.fg_muted)
        text.append("1.02", style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        text.append("Fidelity: ", style=theme.fg_muted)
        text.append("99.2%", style=f"bold {theme.success}")
        text.append("  │  ", style=theme.border)
        
        text.append("Gini: ", style=theme.fg_muted)
        text.append("0.03", style=theme.fg_base)
        
        return text
