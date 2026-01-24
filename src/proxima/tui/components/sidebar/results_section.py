"""Results section for sidebar.

Displays recent results and files.
"""

from textual.widgets import Static
from rich.text import Text

from ...state import TUIState
from ...styles.theme import get_theme
from ...styles.icons import ICON_NEW


class ResultsSection(Static):
    """Recent results section for the sidebar."""
    
    DEFAULT_CSS = """
    ResultsSection {
        height: auto;
        padding: 0;
    }
    """
    
    MAX_VISIBLE = 5
    
    def __init__(self, state: TUIState, **kwargs):
        """Initialize the results section."""
        super().__init__(**kwargs)
        self.state = state
    
    def render(self) -> Text:
        """Render the results section."""
        theme = get_theme()
        text = Text()
        
        # Header
        text.append("Results ", style=f"bold {theme.fg_subtle}")
        text.append("────────────", style=theme.border)
        text.append("\n")
        
        # Get recent results
        results = self.state.result_history[:self.MAX_VISIBLE]
        
        if not results:
            text.append("No results yet", style=theme.fg_subtle)
            return text
        
        for result in results:
            # File name
            file_path = result.file_path
            if file_path:
                # Get just the filename
                filename = file_path.split("/")[-1] if "/" in file_path else file_path
                if len(filename) > 20:
                    filename = filename[:17] + "..."
                
                text.append(f"{filename:<20}", style=theme.fg_base)
                
                # New indicator for recent results
                from datetime import datetime, timedelta
                if result.timestamp > datetime.now() - timedelta(minutes=5):
                    text.append(f" {ICON_NEW}", style=f"bold {theme.success}")
            else:
                text.append(f"{result.id:<20}", style=theme.fg_base)
            
            text.append("\n")
        
        return text
