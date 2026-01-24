"""Command item component for Proxima TUI.

Individual command item in the command palette.
"""

from textual.widgets import ListItem, Label
from rich.text import Text

from ...styles.theme import get_theme


class CommandItem(ListItem):
    """A single command item in the list."""
    
    DEFAULT_CSS = """
    CommandItem {
        padding: 0 1;
        height: 3;
    }
    
    CommandItem:hover {
        background: $primary-darken-3;
    }
    
    CommandItem.-highlighted {
        background: $primary-darken-2;
    }
    """
    
    def __init__(self, command, **kwargs):
        """Initialize the command item.
        
        Args:
            command: The Command object to display
        """
        super().__init__(**kwargs)
        self.command = command
    
    def compose(self):
        """Compose the command item."""
        yield Label(self._render_content())
    
    def _render_content(self) -> Text:
        """Render the command item content."""
        theme = get_theme()
        text = Text()
        
        # Command name
        text.append(self.command.name, style=f"bold {theme.fg_base}")
        
        # Keybinding
        if self.command.keybinding:
            text.append("  ")
            text.append(f"[{self.command.keybinding}]", style=theme.accent)
        
        # Category badge
        text.append("  ")
        text.append(f"({self.command.category})", style=theme.fg_subtle)
        
        text.append("\n")
        
        # Description
        text.append(self.command.description, style=theme.fg_muted)
        
        return text
