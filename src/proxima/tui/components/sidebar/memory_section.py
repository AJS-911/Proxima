"""Memory section for sidebar.

Displays memory usage with visual gauge.
"""

from textual.widgets import Static
from rich.text import Text

from ...state import TUIState
from ...styles.theme import get_theme
from ...styles.icons import PROGRESS_FILLED, PROGRESS_EMPTY, get_memory_indicator


class MemorySection(Static):
    """Memory monitoring section for the sidebar.
    
    Shows memory gauge with threshold indicators.
    """
    
    DEFAULT_CSS = """
    MemorySection {
        height: auto;
        padding: 0;
    }
    """
    
    GAUGE_WIDTH = 10
    
    def __init__(self, state: TUIState, **kwargs):
        """Initialize the memory section."""
        super().__init__(**kwargs)
        self.state = state
    
    def render(self) -> Text:
        """Render the memory section."""
        theme = get_theme()
        text = Text()
        
        # Header
        text.append("Memory ", style=f"bold {theme.fg_subtle}")
        text.append("─────────────", style=theme.border)
        text.append("\n")
        
        # Memory gauge
        percent = self.state.memory_percent
        level = self.state.memory_level
        
        filled = int(self.GAUGE_WIDTH * percent / 100)
        empty = self.GAUGE_WIDTH - filled
        
        # Get color based on level
        color = theme.get_memory_level_color(level)
        
        # Gauge bar
        text.append(PROGRESS_FILLED * filled, style=f"bold {color}")
        text.append(PROGRESS_EMPTY * empty, style=theme.fg_subtle)
        text.append(" ")
        
        # Percentage
        text.append(f"{percent:.0f}%", style=f"bold {color}")
        
        # Level indicator
        indicator = get_memory_indicator(level)
        if indicator:
            text.append(f" {indicator}", style=f"bold {color}")
        else:
            text.append(f" {level}", style=theme.fg_muted)
        
        text.append("\n")
        
        # Absolute values
        text.append(self.state.get_formatted_memory(), style=theme.fg_muted)
        
        return text
