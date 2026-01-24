"""Animated progress bar component.

A smooth, animated progress bar with color transitions.
"""

from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text

from ...styles.theme import get_theme, blend_colors
from ...styles.icons import PROGRESS_FILLED, PROGRESS_EMPTY


class ProgressBar(Static):
    """Animated progress bar with color transitions.
    
    Features:
    - Smooth animation (100 steps)
    - Color transitions: secondary → primary as progress increases
    - Stage name display
    - Percentage with ETA
    """
    
    DEFAULT_CSS = """
    ProgressBar {
        height: 3;
        width: 100%;
        padding: 0 1;
    }
    
    ProgressBar.-compact {
        height: 1;
    }
    """
    
    progress = reactive(0.0)
    stage_name = reactive("")
    eta_text = reactive("")
    
    BAR_WIDTH = 50
    
    def __init__(
        self,
        progress: float = 0.0,
        stage_name: str = "",
        eta_text: str = "",
        compact: bool = False,
        show_percentage: bool = True,
        **kwargs,
    ):
        """Initialize the progress bar.
        
        Args:
            progress: Initial progress (0-100)
            stage_name: Current stage name
            eta_text: ETA text to display
            compact: Use compact single-line mode
            show_percentage: Whether to show percentage
        """
        super().__init__(**kwargs)
        self.progress = progress
        self.stage_name = stage_name
        self.eta_text = eta_text
        self.compact = compact
        self.show_percentage = show_percentage
    
    def render(self) -> Text:
        """Render the progress bar."""
        if self.compact:
            return self._render_compact()
        return self._render_full()
    
    def _render_compact(self) -> Text:
        """Render compact single-line progress bar."""
        theme = get_theme()
        text = Text()
        
        # Calculate filled/empty
        filled = int(self.BAR_WIDTH * self.progress / 100)
        empty = self.BAR_WIDTH - filled
        
        # Get color based on progress
        color = self._get_progress_color()
        
        # Bar
        text.append(PROGRESS_FILLED * filled, style=f"bold {color}")
        text.append(PROGRESS_EMPTY * empty, style=theme.fg_subtle)
        
        # Percentage
        if self.show_percentage:
            text.append(f" {self.progress:>3.0f}%", style=f"bold {color}")
        
        return text
    
    def _render_full(self) -> Text:
        """Render full progress bar with stage info."""
        theme = get_theme()
        text = Text()
        
        # Stage name line
        if self.stage_name:
            text.append(self.stage_name, style=f"bold {theme.fg_base}")
            text.append("\n")
        
        # Progress bar
        filled = int(self.BAR_WIDTH * self.progress / 100)
        empty = self.BAR_WIDTH - filled
        
        color = self._get_progress_color()
        
        text.append(PROGRESS_FILLED * filled, style=f"bold {color}")
        text.append(PROGRESS_EMPTY * empty, style=theme.fg_subtle)
        
        # Percentage
        text.append(f"  {self.progress:>3.0f}%", style=f"bold {color}")
        
        # ETA line
        if self.eta_text:
            text.append("\n")
            text.append(self.eta_text, style=theme.fg_muted)
        
        return text
    
    def _get_progress_color(self) -> str:
        """Get color based on progress percentage."""
        theme = get_theme()
        
        # Blend from secondary (purple) to primary (magenta) as progress increases
        if self.progress < 50:
            return blend_colors(theme.secondary, theme.primary, self.progress / 50)
        else:
            return blend_colors(theme.primary, theme.primary_light, (self.progress - 50) / 50)
    
    def set_progress(self, value: float) -> None:
        """Set the progress value.
        
        Args:
            value: Progress percentage (0-100)
        """
        self.progress = max(0.0, min(100.0, value))
    
    def set_stage(self, name: str) -> None:
        """Set the current stage name.
        
        Args:
            name: Stage name
        """
        self.stage_name = name
    
    def set_eta(self, eta: str) -> None:
        """Set the ETA text.
        
        Args:
            eta: ETA text
        """
        self.eta_text = eta
