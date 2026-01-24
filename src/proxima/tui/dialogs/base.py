"""Base dialog class for Proxima TUI.

Provides common functionality for all dialogs.
"""

from typing import Optional

from textual.screen import ModalScreen
from textual.containers import Vertical
from textual.widgets import Static

from ..styles.theme import get_theme


class BaseDialog(ModalScreen):
    """Base dialog with common styling and functionality.
    
    All dialogs inherit from this to get:
    - Modal overlay behavior
    - Escape to close
    - Consistent styling
    """
    
    DEFAULT_CSS = """
    BaseDialog {
        align: center middle;
    }
    
    BaseDialog > .dialog-container {
        padding: 1 2;
        border: thick $primary;
        background: $surface;
        min-width: 50;
        max-width: 80;
    }
    
    BaseDialog > .dialog-container > .dialog-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }
    
    BaseDialog > .dialog-container > .dialog-content {
        margin: 1 0;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    # Override in subclasses
    DIALOG_TITLE = "Dialog"
    
    def compose(self):
        """Compose the dialog layout."""
        with Vertical(classes="dialog-container"):
            yield Static(self.DIALOG_TITLE, classes="dialog-title")
            yield from self.compose_content()
    
    def compose_content(self):
        """Compose the dialog content.
        
        Override in subclasses to provide dialog-specific content.
        """
        yield Vertical(classes="dialog-content")
    
    def action_cancel(self) -> None:
        """Close the dialog."""
        self.dismiss(None)
