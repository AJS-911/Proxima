"""Model item component for Proxima TUI.

Individual model item in the models dialog.
"""

from dataclasses import dataclass
from typing import Optional

from textual.widgets import ListItem, Label
from rich.text import Text

from ...styles.theme import get_theme
from ...styles.icons import ICON_MODEL, ICON_CONNECTED, ICON_DISCONNECTED


@dataclass
class ModelInfo:
    """Information about an LLM model."""
    
    name: str
    provider: str
    description: str
    available: bool = False
    details: Optional[str] = None


class ModelItem(ListItem):
    """A single model item in the list."""
    
    DEFAULT_CSS = """
    ModelItem {
        height: 3;
        padding: 0 1;
    }
    
    ModelItem:hover {
        background: $primary-darken-3;
    }
    
    ModelItem.-current {
        background: $primary-darken-2;
    }
    """
    
    def __init__(self, model: ModelInfo, is_current: bool = False, **kwargs):
        """Initialize the model item.
        
        Args:
            model: Model information
            is_current: Whether this is the current model
        """
        super().__init__(**kwargs)
        self.model = model
        self.is_current = is_current
        if is_current:
            self.add_class("-current")
    
    def compose(self):
        """Compose the item content."""
        yield Label(self._render_content())
    
    def _render_content(self) -> Text:
        """Render the model item content."""
        theme = get_theme()
        text = Text()
        
        # Status icon
        if self.model.available:
            text.append(ICON_CONNECTED, style=f"bold {theme.success}")
        else:
            text.append(ICON_DISCONNECTED, style=theme.fg_subtle)
        
        text.append(" ")
        
        # Model name
        text.append(self.model.name, style=f"bold {theme.fg_base}")
        
        # Current indicator
        if self.is_current:
            text.append(" (current)", style=theme.accent)
        
        # Provider
        text.append(f"  [{self.model.provider}]", style=theme.fg_muted)
        
        text.append("\n  ")
        
        # Description
        text.append(self.model.description, style=theme.fg_subtle)
        
        # Details
        if self.model.details:
            text.append(f"  •  {self.model.details}", style=theme.fg_subtle)
        
        return text
