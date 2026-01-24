"""Error display dialog for Proxima TUI.

Dialog for displaying errors with details.
"""

from typing import Optional

from textual.screen import ModalScreen
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Button
from rich.text import Text
from rich.traceback import Traceback

from ...styles.theme import get_theme
from ...styles.icons import ICON_ERROR


class ErrorDialog(ModalScreen):
    """Dialog for displaying error information.
    
    Features:
    - Error message display
    - Stack trace (expandable)
    - Copy to clipboard
    - Report issue link
    """
    
    DEFAULT_CSS = """
    ErrorDialog {
        align: center middle;
    }
    
    ErrorDialog > .dialog-container {
        width: 80;
        max-height: 80%;
        border: thick $error;
        background: $surface;
    }
    
    ErrorDialog .dialog-title {
        text-style: bold;
        color: $error;
        text-align: center;
        padding: 1;
        border-bottom: solid $error-darken-2;
    }
    
    ErrorDialog .error-message {
        padding: 1 2;
        color: $text;
    }
    
    ErrorDialog .error-details {
        height: auto;
        max-height: 20;
        margin: 1 2;
        padding: 1;
        border: solid $primary-darken-3;
        background: $surface-darken-1;
        overflow-y: auto;
    }
    
    ErrorDialog .buttons {
        height: auto;
        padding: 1;
        border-top: solid $primary-darken-3;
        layout: horizontal;
        align: center middle;
    }
    
    ErrorDialog .btn {
        margin: 0 1;
    }
    """
    
    BINDINGS = [
        ("escape", "close", "Close"),
        ("c", "copy", "Copy"),
        ("d", "toggle_details", "Details"),
    ]
    
    def __init__(
        self,
        title: str = "Error",
        message: str = "An error occurred",
        details: Optional[str] = None,
        exception: Optional[Exception] = None,
        **kwargs,
    ):
        """Initialize the error dialog.
        
        Args:
            title: Dialog title
            message: Error message
            details: Additional details (stack trace, etc.)
            exception: The exception object
        """
        super().__init__(**kwargs)
        self._title = title
        self.message = message
        self.details = details
        self.exception = exception
        self._details_visible = False
    
    def compose(self):
        """Compose the dialog layout."""
        with Vertical(classes="dialog-container"):
            yield Static(f"{ICON_ERROR} {self._title}", classes="dialog-title")
            yield ErrorMessage(self.message, classes="error-message")
            
            if self.details or self.exception:
                yield ErrorDetails(
                    self.details,
                    self.exception,
                    classes="error-details",
                )
            
            with Horizontal(classes="buttons"):
                yield Button("[C] Copy", id="btn-copy", classes="btn")
                if self.details or self.exception:
                    yield Button("[D] Toggle Details", id="btn-details", classes="btn")
                yield Button("Close", id="btn-close", classes="btn", variant="primary")
    
    def action_close(self) -> None:
        """Close the dialog."""
        self.dismiss()
    
    def action_copy(self) -> None:
        """Copy error to clipboard."""
        import pyperclip
        
        text = f"{self._title}\n\n{self.message}"
        if self.details:
            text += f"\n\n{self.details}"
        
        try:
            pyperclip.copy(text)
            self.notify("Error copied to clipboard")
        except Exception:
            self.notify("Could not copy to clipboard", severity="warning")
    
    def action_toggle_details(self) -> None:
        """Toggle details visibility."""
        self._details_visible = not self._details_visible
        try:
            details = self.query_one(ErrorDetails)
            details.display = self._details_visible
        except Exception:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-close":
            self.action_close()
        elif event.button.id == "btn-copy":
            self.action_copy()
        elif event.button.id == "btn-details":
            self.action_toggle_details()


class ErrorMessage(Static):
    """Main error message display."""
    
    def __init__(self, message: str, **kwargs):
        """Initialize the message display."""
        super().__init__(**kwargs)
        self._message = message
    
    def render(self) -> Text:
        """Render the error message."""
        theme = get_theme()
        text = Text()
        
        text.append(self._message, style=theme.fg_base)
        
        return text


class ErrorDetails(Static):
    """Error details/traceback display."""
    
    def __init__(
        self,
        details: Optional[str],
        exception: Optional[Exception],
        **kwargs,
    ):
        """Initialize the details display."""
        super().__init__(**kwargs)
        self._details = details
        self._exception = exception
        self.display = False  # Hidden by default
    
    def render(self):
        """Render the error details."""
        theme = get_theme()
        
        if self._exception:
            return Traceback.from_exception(
                type(self._exception),
                self._exception,
                self._exception.__traceback__,
            )
        elif self._details:
            text = Text()
            text.append("Details:\n", style=f"bold {theme.fg_muted}")
            text.append(self._details, style=theme.fg_subtle)
            return text
        else:
            return Text("No additional details", style=theme.fg_muted)
