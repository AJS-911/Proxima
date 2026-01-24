"""Input area components for Proxima TUI.

Advanced input widgets with history and autocomplete.
"""

from typing import List, Optional, Callable

from textual.widgets import Input, Static
from textual.containers import Vertical
from textual.message import Message
from textual import on
from rich.text import Text

from ...styles.theme import get_theme


class AutocompleteInput(Input):
    """Input with autocomplete suggestions.
    
    Provides:
    - Tab completion
    - History navigation
    - Suggestion overlay
    """
    
    DEFAULT_CSS = """
    AutocompleteInput {
        border: solid $primary-darken-2;
    }
    
    AutocompleteInput:focus {
        border: solid $primary;
    }
    """
    
    class Submitted(Message):
        """Message sent when input is submitted."""
        def __init__(self, value: str):
            super().__init__()
            self.value = value
    
    def __init__(
        self,
        placeholder: str = "",
        suggestions: Optional[List[str]] = None,
        history: Optional[List[str]] = None,
        on_submit: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        """Initialize the autocomplete input.
        
        Args:
            placeholder: Placeholder text
            suggestions: List of autocomplete suggestions
            history: Command history for up/down navigation
            on_submit: Callback on submit
        """
        super().__init__(placeholder=placeholder, **kwargs)
        self.suggestions = suggestions or []
        self.history = history or []
        self.history_index = -1
        self.on_submit_callback = on_submit
        self._current_suggestions: List[str] = []
    
    def _filter_suggestions(self, prefix: str) -> List[str]:
        """Filter suggestions by prefix."""
        if not prefix:
            return []
        prefix_lower = prefix.lower()
        return [s for s in self.suggestions if s.lower().startswith(prefix_lower)][:5]
    
    def _complete(self) -> None:
        """Complete with the first matching suggestion."""
        if self._current_suggestions:
            self.value = self._current_suggestions[0]
            self.cursor_position = len(self.value)
    
    def _history_up(self) -> None:
        """Navigate up in history."""
        if self.history and self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.value = self.history[-(self.history_index + 1)]
            self.cursor_position = len(self.value)
    
    def _history_down(self) -> None:
        """Navigate down in history."""
        if self.history_index > 0:
            self.history_index -= 1
            self.value = self.history[-(self.history_index + 1)]
            self.cursor_position = len(self.value)
        elif self.history_index == 0:
            self.history_index = -1
            self.value = ""
    
    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed) -> None:
        """Update suggestions on input change."""
        self._current_suggestions = self._filter_suggestions(event.value)
    
    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        value = event.value
        if value:
            # Add to history
            if not self.history or self.history[-1] != value:
                self.history.append(value)
            self.history_index = -1
            
            # Callback
            if self.on_submit_callback:
                self.on_submit_callback(value)
            
            # Post message
            self.post_message(self.Submitted(value))
            
            # Clear input
            self.value = ""
    
    def action_cursor_up(self) -> None:
        """Handle up arrow."""
        self._history_up()
    
    def action_cursor_down(self) -> None:
        """Handle down arrow."""
        self._history_down()
    
    def key_tab(self) -> None:
        """Handle tab key for completion."""
        self._complete()


class InputArea(Vertical):
    """Input area with prompt and status.
    
    Combines:
    - Prompt indicator
    - Autocomplete input
    - Optional status line
    """
    
    DEFAULT_CSS = """
    InputArea {
        height: auto;
        padding: 1;
        border-top: solid $primary-darken-3;
        background: $surface-darken-1;
    }
    
    InputArea .prompt-line {
        layout: horizontal;
        height: auto;
    }
    
    InputArea .prompt-indicator {
        width: auto;
        padding-right: 1;
    }
    
    InputArea .input-field {
        width: 1fr;
    }
    
    InputArea .status-line {
        height: auto;
        margin-top: 1;
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        prompt: str = "❯",
        placeholder: str = "Enter command...",
        show_status: bool = False,
        status_text: str = "",
        **kwargs,
    ):
        """Initialize the input area.
        
        Args:
            prompt: Prompt character/string
            placeholder: Input placeholder
            show_status: Whether to show status line
            status_text: Status line text
        """
        super().__init__(**kwargs)
        self.prompt = prompt
        self.placeholder = placeholder
        self.show_status = show_status
        self.status_text = status_text
    
    def compose(self):
        """Compose the input area."""
        from textual.containers import Horizontal
        
        with Horizontal(classes="prompt-line"):
            yield PromptIndicator(self.prompt, classes="prompt-indicator")
            yield AutocompleteInput(
                placeholder=self.placeholder,
                classes="input-field",
            )
        
        if self.show_status:
            yield Static(self.status_text, classes="status-line")
    
    def get_input(self) -> AutocompleteInput:
        """Get the input widget."""
        return self.query_one(AutocompleteInput)
    
    def set_status(self, text: str) -> None:
        """Update the status text."""
        if self.show_status:
            self.query_one(".status-line").update(text)


class PromptIndicator(Static):
    """Prompt indicator widget."""
    
    def __init__(self, prompt: str = "❯", **kwargs):
        """Initialize the prompt indicator."""
        super().__init__(**kwargs)
        self._prompt = prompt
    
    def render(self) -> Text:
        """Render the prompt."""
        theme = get_theme()
        return Text(self._prompt, style=f"bold {theme.primary}")
