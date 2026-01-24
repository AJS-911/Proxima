"""Result viewer component for Proxima TUI.

Displays simulation results with visualizations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from textual.widgets import Static
from textual.containers import Vertical, Horizontal
from rich.text import Text

from ...styles.theme import get_theme
from ...styles.icons import PROGRESS_FILLED, PROGRESS_EMPTY


@dataclass
class SimulationResult:
    """A simulation result."""
    
    id: str
    task_name: str
    backend: str
    simulator: str
    qubits: int
    shots: int
    execution_time_ms: float
    probabilities: Dict[str, float]
    metadata: Optional[Dict] = None
    
    @property
    def dominant_state(self) -> str:
        """Get the dominant quantum state."""
        if not self.probabilities:
            return "N/A"
        return max(self.probabilities.items(), key=lambda x: x[1])[0]
    
    @property
    def entropy(self) -> float:
        """Calculate Shannon entropy."""
        import math
        entropy = 0.0
        for prob in self.probabilities.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy


class ResultViewer(Vertical):
    """Result viewer with probability distribution.
    
    Features:
    - Header with metadata
    - Probability bar chart
    - Statistics panel
    - Pattern detection
    """
    
    DEFAULT_CSS = """
    ResultViewer {
        height: 100%;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface;
    }
    
    ResultViewer .result-header {
        height: auto;
        margin-bottom: 1;
        padding-bottom: 1;
        border-bottom: solid $primary-darken-3;
    }
    
    ResultViewer .result-title {
        text-style: bold;
        color: $primary;
    }
    
    ResultViewer .result-meta {
        color: $text-muted;
    }
    
    ResultViewer .probability-section {
        height: 1fr;
        margin: 1 0;
        overflow-y: auto;
    }
    
    ResultViewer .stats-section {
        height: auto;
        padding-top: 1;
        border-top: solid $primary-darken-3;
    }
    """
    
    BAR_WIDTH = 40
    
    def __init__(
        self,
        result: Optional[SimulationResult] = None,
        **kwargs,
    ):
        """Initialize the result viewer.
        
        Args:
            result: The simulation result to display
        """
        super().__init__(**kwargs)
        self.result = result
    
    def compose(self):
        """Compose the result viewer."""
        yield ResultHeader(self.result, classes="result-header")
        yield ProbabilityChart(self.result, self.BAR_WIDTH, classes="probability-section")
        yield ResultStats(self.result, classes="stats-section")
    
    def set_result(self, result: SimulationResult) -> None:
        """Set the displayed result."""
        self.result = result
        
        # Update children
        self.query_one(ResultHeader).result = result
        self.query_one(ResultHeader).refresh()
        
        self.query_one(ProbabilityChart).result = result
        self.query_one(ProbabilityChart).refresh()
        
        self.query_one(ResultStats).result = result
        self.query_one(ResultStats).refresh()


class ResultHeader(Static):
    """Header showing result metadata."""
    
    def __init__(self, result: Optional[SimulationResult], **kwargs):
        """Initialize the header."""
        super().__init__(**kwargs)
        self.result = result
    
    def render(self) -> Text:
        """Render the header."""
        theme = get_theme()
        text = Text()
        
        if not self.result:
            text.append("No result loaded", style=theme.fg_muted)
            return text
        
        # Title
        text.append(self.result.task_name, style=f"bold {theme.primary}")
        text.append(f"  (ID: {self.result.id[:8]})", style=theme.fg_subtle)
        text.append("\n")
        
        # Metadata line
        text.append("Backend: ", style=theme.fg_muted)
        text.append(f"{self.result.backend} ({self.result.simulator})", style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        text.append("Qubits: ", style=theme.fg_muted)
        text.append(str(self.result.qubits), style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        text.append("Shots: ", style=theme.fg_muted)
        text.append(str(self.result.shots), style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        text.append("Time: ", style=theme.fg_muted)
        text.append(f"{self.result.execution_time_ms:.1f}ms", style=theme.fg_base)
        
        return text


class ProbabilityChart(Static):
    """Bar chart of probability distribution."""
    
    def __init__(
        self,
        result: Optional[SimulationResult],
        bar_width: int = 40,
        **kwargs,
    ):
        """Initialize the chart."""
        super().__init__(**kwargs)
        self.result = result
        self.bar_width = bar_width
    
    def render(self) -> Text:
        """Render the probability chart."""
        theme = get_theme()
        text = Text()
        
        if not self.result or not self.result.probabilities:
            text.append("No probability data available", style=theme.fg_muted)
            return text
        
        text.append("Probability Distribution:", style=f"bold {theme.fg_base}")
        text.append("\n\n")
        
        # Sort by probability (descending)
        sorted_probs = sorted(
            self.result.probabilities.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        # Show top 10 states
        for state, prob in sorted_probs[:10]:
            # State label (quantum ket notation)
            text.append(f"{state:<10}", style=f"bold {theme.accent}")
            
            # Progress bar
            filled = int(self.bar_width * prob / 100) if prob <= 100 else int(self.bar_width * prob)
            filled = min(filled, self.bar_width)
            empty = self.bar_width - filled
            
            text.append(PROGRESS_FILLED * filled, style=f"bold {theme.primary}")
            text.append(PROGRESS_EMPTY * empty, style=theme.fg_subtle)
            
            # Percentage
            text.append(f" {prob:>6.2f}%", style=theme.fg_muted)
            text.append("\n")
        
        # Show "others" if more than 10 states
        if len(sorted_probs) > 10:
            others_prob = sum(prob for _, prob in sorted_probs[10:])
            text.append("\n")
            text.append(f"... and {len(sorted_probs) - 10} more states ", style=theme.fg_subtle)
            text.append(f"({others_prob:.2f}% total)", style=theme.fg_muted)
        
        return text


class ResultStats(Static):
    """Statistics panel for the result."""
    
    def __init__(self, result: Optional[SimulationResult], **kwargs):
        """Initialize the stats panel."""
        super().__init__(**kwargs)
        self.result = result
    
    def render(self) -> Text:
        """Render the statistics."""
        theme = get_theme()
        text = Text()
        
        if not self.result:
            return text
        
        # Entropy
        text.append("Entropy: ", style=theme.fg_muted)
        entropy = self.result.entropy
        text.append(f"{entropy:.3f} bits", style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        # Dominant state
        text.append("Dominant: ", style=theme.fg_muted)
        text.append(self.result.dominant_state, style=f"bold {theme.accent}")
        text.append("  │  ", style=theme.border)
        
        # Number of states
        text.append("States: ", style=theme.fg_muted)
        text.append(str(len(self.result.probabilities)), style=theme.fg_base)
        
        # Pattern detection (simple heuristic)
        text.append("\n")
        text.append("Pattern: ", style=theme.fg_muted)
        pattern = self._detect_pattern()
        if pattern:
            text.append(pattern, style=f"bold {theme.success}")
        else:
            text.append("Unknown", style=theme.fg_subtle)
        
        return text
    
    def _detect_pattern(self) -> Optional[str]:
        """Simple pattern detection heuristic."""
        if not self.result or not self.result.probabilities:
            return None
        
        probs = self.result.probabilities
        states = list(probs.keys())
        
        # Check for Bell state pattern (|00> + |11> or similar)
        if len(states) == 2:
            prob_values = list(probs.values())
            if all(45 <= p <= 55 for p in prob_values):
                return "Bell State"
        
        # Check for GHZ state pattern (|000...> + |111...>)
        if len(states) == 2:
            if all(all(c == '0' for c in s) or all(c == '1' for c in s) for s in states):
                prob_values = list(probs.values())
                if all(45 <= p <= 55 for p in prob_values):
                    return "GHZ State"
        
        # Check for uniform distribution (Hadamard on all qubits)
        if len(states) > 1:
            avg_prob = 100.0 / len(states)
            if all(abs(p - avg_prob) < 5 for p in probs.values()):
                return "Uniform Superposition"
        
        return None
