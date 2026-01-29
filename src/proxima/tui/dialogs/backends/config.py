"""Backend configuration dialog for managing backend settings."""

from textual.screen import ModalScreen
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, Button, Input, Switch, Select, Label
from typing import Dict, Any, Optional

try:
    from proxima.backends.registry import BackendRegistry
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False


class BackendConfigDialog(ModalScreen):
    """Dialog for configuring backend settings."""

    DEFAULT_CSS = """
    BackendConfigDialog { align: center middle; }
    BackendConfigDialog > .dialog-container {
        padding: 1 2;
        border: thick $accent;
        background: $surface;
        width: 100%;
        height: 100%;
        max-width: 100%;
        max-height: 100%;
    }
    BackendConfigDialog .dialog-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
        height: 2;
    }
    BackendConfigDialog ScrollableContainer {
        height: 1fr;
    }
    BackendConfigDialog .config-section {
        margin: 1 0;
        padding: 1;
        border: solid $primary-darken-2;
        height: auto;
    }
    BackendConfigDialog .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        height: 1;
    }
    BackendConfigDialog .config-row {
        layout: horizontal;
        height: 3;
        margin: 0 0 1 0;
        align: left middle;
    }
    BackendConfigDialog .config-label {
        width: 20;
        height: 3;
        content-align: left middle;
    }
    BackendConfigDialog .config-input {
        width: 1fr;
        min-width: 20;
    }
    BackendConfigDialog Input {
        width: 100%;
        height: 3;
    }
    BackendConfigDialog Select {
        width: 100%;
        height: 3;
    }
    BackendConfigDialog Switch {
        margin-left: 1;
    }
    BackendConfigDialog .footer {
        height: auto;
        layout: horizontal;
        margin-top: 1;
    }
    BackendConfigDialog Button {
        margin-right: 1;
    }
    """

    BINDINGS = [
        ("escape", "close", "Close"),
        ("ctrl+s", "save", "Save"),
    ]

    def __init__(self, backend_name: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.backend_name = backend_name or "Default"
        self._config: Dict[str, Any] = {}
        self._original_config: Dict[str, Any] = {}

    def compose(self):
        with Vertical(classes="dialog-container"):
            yield Static(f"[*] Configure: {self.backend_name}", classes="dialog-title")
            
            with ScrollableContainer():
                # General Settings
                with Vertical(classes="config-section"):
                    yield Static("General Settings", classes="section-title")
                    
                    with Horizontal(classes="config-row"):
                        yield Label("Timeout (ms):", classes="config-label")
                        yield Input(value="5000", id="config-timeout", classes="config-input", type="integer")
                    
                    with Horizontal(classes="config-row"):
                        yield Label("Max Retries:", classes="config-label")
                        yield Input(value="3", id="config-retries", classes="config-input", type="integer")
                    
                    with Horizontal(classes="config-row"):
                        yield Label("Auto-connect:", classes="config-label")
                        yield Switch(value=True, id="config-autoconnect")
                
                # Performance Settings
                with Vertical(classes="config-section"):
                    yield Static("Performance Settings", classes="section-title")
                    
                    with Horizontal(classes="config-row"):
                        yield Label("Thread Pool:", classes="config-label")
                        yield Input(value="4", id="config-threads", classes="config-input", type="integer")
                    
                    with Horizontal(classes="config-row"):
                        yield Label("Cache Size:", classes="config-label")
                        yield Input(value="1024", id="config-cache", classes="config-input", type="integer")
                    
                    with Horizontal(classes="config-row"):
                        yield Label("Enable Cache:", classes="config-label")
                        yield Switch(value=True, id="config-cache-enabled")
                
                # Advanced Settings
                with Vertical(classes="config-section"):
                    yield Static("Advanced Settings", classes="section-title")
                    
                    with Horizontal(classes="config-row"):
                        yield Label("Log Level:", classes="config-label")
                        yield Select(
                            [(level, level) for level in ["DEBUG", "INFO", "WARNING", "ERROR"]],
                            value="INFO",
                            id="config-loglevel"
                        )
                    
                    with Horizontal(classes="config-row"):
                        yield Label("Validation:", classes="config-label")
                        yield Switch(value=True, id="config-validation")
            
            with Horizontal(classes="footer"):
                yield Button("[S] Save", id="btn-save", variant="success")
                yield Button("[R] Reset", id="btn-reset", variant="warning")
                yield Button("[X] Cancel", id="btn-cancel", variant="error")

    def on_mount(self):
        """Load current configuration."""
        self._load_config()

    def _load_config(self):
        """Load configuration from registry or use defaults."""
        self._config = {
            "timeout": 5000,
            "retries": 3,
            "autoconnect": True,
            "threads": 4,
            "cache_size": 1024,
            "cache_enabled": True,
            "log_level": "INFO",
            "validation": True,
        }
        
        if REGISTRY_AVAILABLE and self.backend_name != "Default":
            try:
                registry = BackendRegistry()
                info = registry.get_backend_info(self.backend_name)
                if info and "config" in info:
                    self._config.update(info["config"])
            except Exception:
                pass
        
        self._original_config = self._config.copy()
        self._apply_config_to_ui()

    def _apply_config_to_ui(self):
        """Apply loaded config to UI elements."""
        try:
            self.query_one("#config-timeout", Input).value = str(self._config.get("timeout", 5000))
            self.query_one("#config-retries", Input).value = str(self._config.get("retries", 3))
            self.query_one("#config-autoconnect", Switch).value = self._config.get("autoconnect", True)
            self.query_one("#config-threads", Input).value = str(self._config.get("threads", 4))
            self.query_one("#config-cache", Input).value = str(self._config.get("cache_size", 1024))
            self.query_one("#config-cache-enabled", Switch).value = self._config.get("cache_enabled", True)
            self.query_one("#config-validation", Switch).value = self._config.get("validation", True)
        except Exception:
            pass

    def _read_config_from_ui(self) -> Dict[str, Any]:
        """Read current config values from UI elements."""
        try:
            return {
                "timeout": int(self.query_one("#config-timeout", Input).value or 5000),
                "retries": int(self.query_one("#config-retries", Input).value or 3),
                "autoconnect": self.query_one("#config-autoconnect", Switch).value,
                "threads": int(self.query_one("#config-threads", Input).value or 4),
                "cache_size": int(self.query_one("#config-cache", Input).value or 1024),
                "cache_enabled": self.query_one("#config-cache-enabled", Switch).value,
                "log_level": self.query_one("#config-loglevel", Select).value,
                "validation": self.query_one("#config-validation", Switch).value,
            }
        except Exception:
            return self._config

    def on_button_pressed(self, event):
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "btn-cancel":
            self.dismiss(None)
        elif button_id == "btn-save":
            self.action_save()
        elif button_id == "btn-reset":
            self._config = self._original_config.copy()
            self._apply_config_to_ui()
            self.notify("Configuration reset to original values")

    def action_close(self):
        """Close dialog without saving."""
        self.dismiss(None)

    def action_save(self):
        """Save configuration."""
        self._config = self._read_config_from_ui()
        
        # Validate
        if self._config["timeout"] < 100:
            self.notify("Timeout must be at least 100ms", severity="error")
            return
        if self._config["retries"] < 0:
            self.notify("Retries cannot be negative", severity="error")
            return
        if self._config["threads"] < 1:
            self.notify("Thread pool must be at least 1", severity="error")
            return
        
        # Save to file
        try:
            import json
            from pathlib import Path
            
            config_dir = Path.home() / ".proxima"
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / f"backend_{self.backend_name.lower()}.json"
            
            with open(config_file, "w") as f:
                json.dump(self._config, f, indent=2)
            
            self.notify(f"[+] Configuration saved", severity="success")
            self.dismiss(self._config)
        except Exception as e:
            self.notify(f"Save failed: {e}", severity="error")
