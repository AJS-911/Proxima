"""Backends section for sidebar.

Displays backend health status and availability.
"""

from textual.widgets import Static
from rich.text import Text

from ...state import TUIState
from ...styles.theme import get_theme
from ...styles.icons import get_health_icon, ELLIPSIS


class BackendsSection(Static):
    """Backend status section for the sidebar.
    
    Shows health indicators for each backend.
    """
    
    DEFAULT_CSS = """
    BackendsSection {
        height: auto;
        padding: 0;
    }
    """
    
    MAX_VISIBLE = 5
    
    def __init__(self, state: TUIState, **kwargs):
        """Initialize the backends section."""
        super().__init__(**kwargs)
        self.state = state
    
    def render(self) -> Text:
        """Render the backends section."""
        theme = get_theme()
        text = Text()
        
        # Header
        text.append("Backends ", style=f"bold {theme.fg_subtle}")
        text.append("───────────", style=theme.border)
        text.append("\n")
        
        # Get backend statuses
        backends = list(self.state.backend_statuses.items())
        
        if not backends:
            text.append("No backends configured", style=theme.fg_subtle)
            return text
        
        # Show first N backends
        visible = backends[:self.MAX_VISIBLE]
        hidden_count = len(backends) - self.MAX_VISIBLE
        
        for name, status in visible:
            # Health icon
            icon = get_health_icon(status.status)
            color = theme.get_health_color(status.status)
            
            text.append(icon, style=f"bold {color}")
            text.append(" ")
            
            # Backend name (padded)
            display_name = name.capitalize()
            text.append(f"{display_name:<12}", style=theme.fg_base)
            
            # Status
            status_text = status.status.lower()
            if status.status == "healthy":
                status_text = "healthy"
            elif status.status == "unknown":
                status_text = "unavail"
            
            text.append(f"{status_text:<10}", style=theme.fg_muted)
            
            # Response time (if healthy)
            if status.response_time_ms and status.status == "healthy":
                text.append(f"{status.response_time_ms:.0f}ms", style=theme.fg_subtle)
            
            text.append("\n")
        
        # Show "and N more" if there are hidden backends
        if hidden_count > 0:
            text.append(f"… and {hidden_count} more", style=theme.fg_subtle)
        
        return text
