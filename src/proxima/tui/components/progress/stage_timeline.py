"""Stage timeline component.

Displays execution stages with completion status.
"""

from typing import List, Optional

from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text

from ...state.tui_state import StageInfo
from ...styles.theme import get_theme
from ...styles.icons import get_stage_icon


class StageTimeline(Static):
    """Execution stage timeline display.
    
    Shows stages with:
    - ✓ Completed (success color)
    - ● Current (primary color with animation)
    - ○ Pending (muted color)
    - ✗ Failed (error color)
    """
    
    DEFAULT_CSS = """
    StageTimeline {
        height: auto;
        width: 100%;
        padding: 0 1;
    }
    """
    
    def __init__(
        self,
        stages: Optional[List[StageInfo]] = None,
        current_index: int = 0,
        show_times: bool = True,
        show_eta: bool = True,
        total_elapsed_ms: float = 0.0,
        total_eta_ms: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the stage timeline.
        
        Args:
            stages: List of stage info
            current_index: Current stage index
            show_times: Whether to show durations
            show_eta: Whether to show ETA
            total_elapsed_ms: Total elapsed time
            total_eta_ms: Total estimated time
        """
        super().__init__(**kwargs)
        self._stages = stages or []
        self._current_index = current_index
        self.show_times = show_times
        self.show_eta = show_eta
        self.total_elapsed_ms = total_elapsed_ms
        self.total_eta_ms = total_eta_ms
    
    def render(self) -> Text:
        """Render the stage timeline."""
        theme = get_theme()
        text = Text()
        
        if not self._stages:
            text.append("No stages", style=theme.fg_subtle)
            return text
        
        for i, stage in enumerate(self._stages):
            # Determine stage status
            if stage.status == "done":
                icon = get_stage_icon("done")
                color = theme.success
                time_text = self._format_duration(stage.duration_ms)
            elif stage.status == "error":
                icon = get_stage_icon("error")
                color = theme.error
                time_text = self._format_duration(stage.duration_ms) if stage.duration_ms else "--"
            elif i == self._current_index or stage.status == "current":
                icon = get_stage_icon("current")
                color = theme.primary
                # Show elapsed + ETA for current stage
                if stage.duration_ms > 0:
                    time_text = f"{self._format_duration(stage.duration_ms)}"
                    if self.show_eta and self.total_eta_ms:
                        remaining = max(0, self.total_eta_ms - self.total_elapsed_ms)
                        time_text += f" (ETA: {self._format_duration(remaining)})"
                else:
                    time_text = "--"
            else:
                icon = get_stage_icon("pending")
                color = theme.fg_subtle
                time_text = "--"
            
            # Icon
            text.append(icon, style=f"bold {color}")
            text.append(" ")
            
            # Stage name (padded)
            name = stage.name[:20] if len(stage.name) > 20 else stage.name
            text.append(f"{name:<22}", style=color)
            
            # Duration
            if self.show_times:
                text.append(time_text, style=theme.fg_muted)
            
            text.append("\n")
        
        # Total line
        if self.show_times:
            text.append("\n")
            elapsed = self._format_duration(self.total_elapsed_ms)
            text.append(f"Total Elapsed: {elapsed}", style=theme.fg_muted)
            
            if self.show_eta and self.total_eta_ms:
                total = self._format_duration(self.total_eta_ms)
                text.append(f"  │  Estimated Total: {total}", style=theme.fg_muted)
        
        return text
    
    def _format_duration(self, ms: float) -> str:
        """Format duration in milliseconds to human readable string."""
        if ms < 0:
            return "--"
        
        seconds = ms / 1000
        
        if seconds < 1:
            return f"{ms:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
    
    def set_stages(self, stages: List[StageInfo]) -> None:
        """Set the stages list.
        
        Args:
            stages: List of stage info
        """
        self._stages = stages
        self.refresh()
    
    def set_current_index(self, index: int) -> None:
        """Set the current stage index.
        
        Args:
            index: Current stage index
        """
        self._current_index = index
        self.refresh()
    
    def update_stage(self, index: int, status: str, duration_ms: float = 0.0) -> None:
        """Update a stage's status.
        
        Args:
            index: Stage index
            status: New status
            duration_ms: Stage duration
        """
        if 0 <= index < len(self._stages):
            self._stages[index].status = status
            self._stages[index].duration_ms = duration_ms
            self.refresh()
    
    def set_totals(self, elapsed_ms: float, eta_ms: Optional[float] = None) -> None:
        """Set total timing values.
        
        Args:
            elapsed_ms: Total elapsed time
            eta_ms: Total estimated time
        """
        self.total_elapsed_ms = elapsed_ms
        self.total_eta_ms = eta_ms
        self.refresh()
