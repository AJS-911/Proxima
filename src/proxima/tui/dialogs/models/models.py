"""LLM model selection dialog for Proxima TUI.

Dialog for selecting and configuring LLM models.
"""

from dataclasses import dataclass
from typing import List, Optional

from textual.screen import ModalScreen
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Input, ListView, ListItem, Switch
from textual import on
from rich.text import Text

from ...styles.theme import get_theme
from ...styles.icons import ICON_MODEL, ICON_CONNECTED, ICON_DISCONNECTED
from .models_item import ModelItem, ModelInfo


# Default available models
DEFAULT_MODELS = [
    ModelInfo("llama2", "Ollama", "Local LLM", True, "7B parameters"),
    ModelInfo("codellama", "Ollama", "Code-focused LLM", True, "7B parameters"),
    ModelInfo("mistral", "Ollama", "Fast local model", False, "7B parameters"),
    ModelInfo("gpt-4", "OpenAI", "Most capable model", False, "API key required"),
    ModelInfo("gpt-3.5-turbo", "OpenAI", "Fast and affordable", False, "API key required"),
    ModelInfo("claude-3-opus", "Anthropic", "Most intelligent", False, "API key required"),
    ModelInfo("claude-3-sonnet", "Anthropic", "Balanced", False, "API key required"),
]


class ModelsDialog(ModalScreen):
    """Dialog for selecting LLM models.
    
    Features:
    - List available models by provider
    - Show model status (available/unavailable)
    - Toggle thinking mode
    - Search/filter models
    """
    
    DEFAULT_CSS = """
    ModelsDialog {
        align: center middle;
    }
    
    ModelsDialog > .dialog-container {
        width: 70;
        height: 30;
        border: thick $primary;
        background: $surface;
    }
    
    ModelsDialog .dialog-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        padding: 1;
        border-bottom: solid $primary-darken-3;
    }
    
    ModelsDialog .search-input {
        margin: 1;
        border: solid $primary-darken-2;
    }
    
    ModelsDialog .models-list {
        height: 1fr;
        margin: 0 1;
    }
    
    ModelsDialog .options-section {
        height: auto;
        padding: 1;
        border-top: solid $primary-darken-3;
    }
    
    ModelsDialog .footer {
        height: auto;
        padding: 1;
        border-top: solid $primary-darken-3;
        text-align: center;
    }
    """
    
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "select", "Select"),
        ("up", "move_up", "Up"),
        ("down", "move_down", "Down"),
        ("t", "toggle_thinking", "Toggle Thinking"),
    ]
    
    def __init__(
        self,
        models: Optional[List[ModelInfo]] = None,
        current_model: Optional[str] = None,
        thinking_enabled: bool = False,
        **kwargs,
    ):
        """Initialize the models dialog.
        
        Args:
            models: Available models (uses defaults if None)
            current_model: Currently selected model name
            thinking_enabled: Whether thinking mode is enabled
        """
        super().__init__(**kwargs)
        self.models = models or DEFAULT_MODELS
        self.filtered_models = self.models.copy()
        self.current_model = current_model
        self.thinking_enabled = thinking_enabled
        self.selected_index = 0
    
    def compose(self):
        """Compose the dialog layout."""
        with Vertical(classes="dialog-container"):
            yield Static(f"{ICON_MODEL} Select Model", classes="dialog-title")
            yield Input(
                placeholder="Search models...",
                classes="search-input",
            )
            yield ModelsListView(self.filtered_models, self.current_model, classes="models-list")
            
            with Horizontal(classes="options-section"):
                yield Static("Enable Thinking Mode: ", classes="option-label")
                yield Switch(value=self.thinking_enabled, id="thinking-switch")
            
            yield ModelsFooter(classes="footer")
    
    def on_mount(self) -> None:
        """Focus search on mount."""
        self.query_one(Input).focus()
    
    @on(Input.Changed)
    def filter_models(self, event: Input.Changed) -> None:
        """Filter models by search query."""
        query = event.value.lower()
        
        if query:
            self.filtered_models = [
                m for m in self.models
                if query in m.name.lower() or
                   query in m.provider.lower() or
                   query in m.description.lower()
            ]
        else:
            self.filtered_models = self.models.copy()
        
        # Update list
        models_list = self.query_one(ModelsListView)
        models_list.update_models(self.filtered_models)
        self.selected_index = 0
    
    def action_cancel(self) -> None:
        """Cancel and close."""
        self.dismiss(None)
    
    def action_select(self) -> None:
        """Select current model."""
        if self.filtered_models and self.selected_index < len(self.filtered_models):
            model = self.filtered_models[self.selected_index]
            thinking = self.query_one("#thinking-switch", Switch).value
            self.dismiss({"model": model, "thinking": thinking})
    
    def action_move_up(self) -> None:
        """Move selection up."""
        if self.selected_index > 0:
            self.selected_index -= 1
            self.query_one(ModelsListView).index = self.selected_index
    
    def action_move_down(self) -> None:
        """Move selection down."""
        if self.selected_index < len(self.filtered_models) - 1:
            self.selected_index += 1
            self.query_one(ModelsListView).index = self.selected_index
    
    def action_toggle_thinking(self) -> None:
        """Toggle thinking mode."""
        switch = self.query_one("#thinking-switch", Switch)
        switch.value = not switch.value


class ModelsListView(ListView):
    """List view for models."""
    
    def __init__(self, models: List[ModelInfo], current: Optional[str], **kwargs):
        """Initialize the list."""
        super().__init__(**kwargs)
        self._models = models
        self._current = current
    
    def on_mount(self) -> None:
        """Populate the list."""
        self.update_models(self._models)
    
    def update_models(self, models: List[ModelInfo]) -> None:
        """Update the displayed models."""
        self.clear()
        for model in models:
            is_current = model.name == self._current
            self.append(ModelItem(model, is_current))
        self._models = models


class ModelsFooter(Static):
    """Footer with keybindings."""
    
    def render(self) -> Text:
        """Render the footer."""
        theme = get_theme()
        text = Text()
        
        bindings = [
            ("↑↓", "navigate"),
            ("enter", "select"),
            ("t", "toggle thinking"),
            ("esc", "cancel"),
        ]
        
        for i, (key, desc) in enumerate(bindings):
            if i > 0:
                text.append(" │ ", style=theme.border)
            text.append(key, style=f"bold {theme.accent}")
            text.append(f" {desc}", style=theme.fg_muted)
        
        return text
