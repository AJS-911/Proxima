"""Session section for sidebar.

Displays current session/task information.
"""

from textual.widgets import Static
from rich.text import Text

from ...state import TUIState
from ...styles.theme import get_theme


class SessionSection(Static):
    """Session/task information section for the sidebar."""
    
    DEFAULT_CSS = """
    SessionSection {
        height: auto;
        padding: 0;
        margin-bottom: 1;
    }
    
    SessionSection.-hidden {
        display: none;
    }
    """
    
    def __init__(self, state: TUIState, **kwargs):
        """Initialize the session section."""
        super().__init__(**kwargs)
        self.state = state
    
    def render(self) -> Text:
        """Render the session section."""
        theme = get_theme()
        text = Text()
        
        # Task title
        task = self.state.current_task or self.state.session_title
        if task:
            text.append(task, style=f"bold {theme.fg_base}")
            text.append("\n")
            
            # Working directory
            if self.state.working_directory:
                wd = self.state.working_directory
                # Shorten path if too long
                if len(wd) > 28:
                    wd = "~/" + wd.split("/")[-1] if "/" in wd else wd[-25:]
                text.append(wd, style=theme.fg_muted)
        else:
            # No active session
            text.append("No active session", style=theme.fg_subtle)
        
        return text
    
    def _update_visibility(self) -> None:
        """Update visibility based on session state."""
        has_session = bool(self.state.current_task or self.state.session_title)
        self.set_class(not has_session, "-hidden")
