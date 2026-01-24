"""Input dialog for Proxima TUI.

Simple text input dialogs.
"""

from typing import Optional

from textual.screen import ModalScreen
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Input, Button
from textual import on

from ...styles.theme import get_theme


class InputDialog(ModalScreen):
    """Simple text input dialog.
    
    Used for prompting user for text input.
    """
    
    DEFAULT_CSS = """
    InputDialog {
        align: center middle;
    }
    
    InputDialog > .dialog-container {
        width: 60;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    
    InputDialog .dialog-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }
    
    InputDialog .dialog-prompt {
        margin-bottom: 1;
        color: $text-muted;
    }
    
    InputDialog .input-field {
        margin: 1 0;
        border: solid $primary-darken-2;
    }
    
    InputDialog .input-field:focus {
        border: solid $primary;
    }
    
    InputDialog .buttons {
        layout: horizontal;
        height: auto;
        margin-top: 1;
        align: center middle;
    }
    
    InputDialog .btn {
        margin: 0 1;
        min-width: 12;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]
    
    def __init__(
        self,
        title: str = "Input",
        prompt: str = "Enter a value:",
        placeholder: str = "",
        default_value: str = "",
        submit_label: str = "OK",
        cancel_label: str = "Cancel",
        validator: Optional[callable] = None,
        **kwargs,
    ):
        """Initialize the input dialog.
        
        Args:
            title: Dialog title
            prompt: Input prompt
            placeholder: Input placeholder text
            default_value: Default input value
            submit_label: Label for submit button
            cancel_label: Label for cancel button
            validator: Optional validation function
        """
        super().__init__(**kwargs)
        self.title_text = title
        self.prompt = prompt
        self.placeholder = placeholder
        self.default_value = default_value
        self.submit_label = submit_label
        self.cancel_label = cancel_label
        self.validator = validator
    
    def compose(self):
        """Compose the dialog layout."""
        with Vertical(classes="dialog-container"):
            yield Static(self.title_text, classes="dialog-title")
            yield Static(self.prompt, classes="dialog-prompt")
            yield Input(
                value=self.default_value,
                placeholder=self.placeholder,
                classes="input-field",
            )
            
            with Horizontal(classes="buttons"):
                yield Button(
                    self.submit_label,
                    id="btn-submit",
                    classes="btn",
                    variant="primary",
                )
                yield Button(
                    self.cancel_label,
                    id="btn-cancel",
                    classes="btn",
                )
    
    def on_mount(self) -> None:
        """Focus the input on mount."""
        self.query_one(Input).focus()
    
    @on(Input.Submitted)
    def submit_input(self, event: Input.Submitted) -> None:
        """Submit on Enter key."""
        self._submit()
    
    def action_cancel(self) -> None:
        """Cancel the dialog."""
        self.dismiss(None)
    
    def _submit(self) -> None:
        """Submit the input value."""
        value = self.query_one(Input).value
        
        if self.validator:
            try:
                self.validator(value)
            except ValueError as e:
                self.notify(str(e), severity="error")
                return
        
        self.dismiss(value)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-submit":
            self._submit()
        else:
            self.action_cancel()
