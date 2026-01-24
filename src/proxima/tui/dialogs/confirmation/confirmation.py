"""Confirmation dialog for Proxima TUI.

Simple yes/no confirmation dialogs.
"""

from textual.screen import ModalScreen
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Button

from ...styles.theme import get_theme


class ConfirmationDialog(ModalScreen):
    """Simple confirmation dialog.
    
    Used for confirming destructive actions.
    """
    
    DEFAULT_CSS = """
    ConfirmationDialog {
        align: center middle;
    }
    
    ConfirmationDialog > .dialog-container {
        width: 50;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    
    ConfirmationDialog .dialog-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }
    
    ConfirmationDialog .dialog-message {
        margin: 1 0;
        text-align: center;
    }
    
    ConfirmationDialog .buttons {
        layout: horizontal;
        height: auto;
        margin-top: 1;
        align: center middle;
    }
    
    ConfirmationDialog .btn {
        margin: 0 1;
        min-width: 12;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("y", "confirm", "Yes"),
        ("n", "cancel", "No"),
    ]
    
    def __init__(
        self,
        title: str = "Confirm",
        message: str = "Are you sure?",
        confirm_label: str = "Yes",
        cancel_label: str = "No",
        destructive: bool = False,
        **kwargs,
    ):
        """Initialize the confirmation dialog.
        
        Args:
            title: Dialog title
            message: Confirmation message
            confirm_label: Label for confirm button
            cancel_label: Label for cancel button
            destructive: Whether this is a destructive action
        """
        super().__init__(**kwargs)
        self.title_text = title
        self.message = message
        self.confirm_label = confirm_label
        self.cancel_label = cancel_label
        self.destructive = destructive
    
    def compose(self):
        """Compose the dialog layout."""
        with Vertical(classes="dialog-container"):
            yield Static(self.title_text, classes="dialog-title")
            yield Static(self.message, classes="dialog-message")
            
            with Horizontal(classes="buttons"):
                yield Button(
                    f"[Y] {self.confirm_label}",
                    id="btn-confirm",
                    classes="btn",
                    variant="error" if self.destructive else "success",
                )
                yield Button(
                    f"[N] {self.cancel_label}",
                    id="btn-cancel",
                    classes="btn",
                )
    
    def action_confirm(self) -> None:
        """Confirm the action."""
        self.dismiss(True)
    
    def action_cancel(self) -> None:
        """Cancel the action."""
        self.dismiss(False)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-confirm":
            self.action_confirm()
        else:
            self.action_cancel()
