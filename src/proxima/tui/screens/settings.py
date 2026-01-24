"""Settings screen for Proxima TUI.

Configuration management.
"""

from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Button, Input, Switch, Select, OptionList
from textual.widgets.option_list import Option
from rich.text import Text

from .base import BaseScreen
from ..styles.theme import get_theme


class SettingsScreen(BaseScreen):
    """Configuration settings screen.
    
    Shows:
    - General settings
    - Backend configuration
    - LLM settings
    - Display preferences
    """
    
    SCREEN_NAME = "settings"
    SCREEN_TITLE = "Configuration"
    
    DEFAULT_CSS = """
    SettingsScreen .settings-container {
        padding: 1;
    }
    
    SettingsScreen .settings-section {
        margin-bottom: 2;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface;
    }
    
    SettingsScreen .section-title {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    SettingsScreen .setting-row {
        height: auto;
        layout: horizontal;
        margin-bottom: 1;
    }
    
    SettingsScreen .setting-label {
        width: 25;
        color: $text-muted;
    }
    
    SettingsScreen .setting-value {
        width: 1fr;
    }
    
    SettingsScreen .setting-input {
        width: 30;
    }
    
    SettingsScreen .actions-section {
        height: auto;
        layout: horizontal;
        margin-top: 1;
    }
    
    SettingsScreen .action-btn {
        margin-right: 1;
    }
    """
    
    def compose_main(self):
        """Compose the settings screen content."""
        with Vertical(classes="main-content settings-container"):
            # General Settings
            with Container(classes="settings-section"):
                yield Static("General Settings", classes="section-title")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Default Backend:", classes="setting-label")
                    yield Static("Cirq", classes="setting-value", id="backend-value")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Default Shots:", classes="setting-label")
                    yield Input(value="1024", classes="setting-input")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Auto-save Results:", classes="setting-label")
                    yield Switch(value=True)
            
            # LLM Settings
            with Container(classes="settings-section"):
                yield Static("LLM Configuration", classes="section-title")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Provider:", classes="setting-label")
                    yield Static("Ollama (Local)", classes="setting-value", id="provider-value")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Model:", classes="setting-label")
                    yield Input(value="llama2", classes="setting-input")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Enable Thinking:", classes="setting-label")
                    yield Switch(value=False)
            
            # Display Settings
            with Container(classes="settings-section"):
                yield Static("Display Settings", classes="section-title")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Theme:", classes="setting-label")
                    yield Static("Dark", classes="setting-value", id="theme-value")
                
                with Horizontal(classes="setting-row"):
                    yield Static("Compact Sidebar:", classes="setting-label")
                    yield Switch(value=False)
                
                with Horizontal(classes="setting-row"):
                    yield Static("Show Log Panel:", classes="setting-label")
                    yield Switch(value=True)
            
            # Actions
            with Horizontal(classes="actions-section"):
                yield Button("Save Settings", id="btn-save", classes="action-btn", variant="primary")
                yield Button("Reset to Defaults", id="btn-reset", classes="action-btn")
                yield Button("Export Config", id="btn-export", classes="action-btn")
                yield Button("Import Config", id="btn-import", classes="action-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-save":
            self.notify("Settings saved!", severity="success")
        elif button_id == "btn-reset":
            self.notify("Settings reset to defaults")
        elif button_id == "btn-export":
            self.notify("Exporting configuration...")
        elif button_id == "btn-import":
            self.notify("Import configuration file...")
