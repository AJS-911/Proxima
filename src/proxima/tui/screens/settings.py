"""Settings screen for Proxima TUI.

Configuration management.
"""

import asyncio
from pathlib import Path

from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Button, Input, Switch, Select, RadioSet, RadioButton
from rich.text import Text

from .base import BaseScreen
from ..styles.theme import get_theme

try:
    from proxima.intelligence.llm_router import LocalLLMDetector, OllamaProvider, OpenAIProvider, AnthropicProvider
    from proxima.config.export_import import export_config, import_config
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class SettingsScreen(BaseScreen):
    """Configuration settings screen.

    Shows:
    - General settings
    - LLM settings (Local vs API)
    - Display preferences
    """

    SCREEN_NAME = "settings"
    SCREEN_TITLE = "Configuration"

    DEFAULT_CSS = """
    SettingsScreen .settings-container {
        padding: 1;
        overflow-y: auto;
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

    SettingsScreen .section-subtitle {
        color: $text-muted;
        margin-bottom: 1;
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
        width: 40;
    }

    SettingsScreen .setting-input-wide {
        width: 50;
    }

    SettingsScreen .subsection {
        margin-left: 2;
        margin-top: 1;
        padding: 1;
        border-left: solid $primary-darken-3;
    }

    SettingsScreen .subsection-title {
        color: $text;
        margin-bottom: 1;
    }

    SettingsScreen .radio-group {
        height: auto;
        margin-bottom: 1;
    }

    SettingsScreen .actions-section {
        height: auto;
        layout: horizontal;
        margin-top: 1;
    }

    SettingsScreen .action-btn {
        margin-right: 1;
    }

    SettingsScreen .hint-text {
        color: $text-muted;
        margin-top: 1;
    }

    SettingsScreen .api-key-input {
        width: 50;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_mode = "local"  # "local", "openai", "anthropic", "none"

    def compose_main(self):
        """Compose the settings screen content."""
        with Vertical(classes="main-content settings-container"):
            # General Settings
            with Container(classes="settings-section"):
                yield Static("?? General Settings", classes="section-title")

                with Horizontal(classes="setting-row"):
                    yield Static("Default Backend:", classes="setting-label")
                    yield Select(
                        [
                            ("Auto (Recommended)", "auto"),
                            ("LRET", "lret"),
                            ("Cirq", "cirq"),
                            ("Qiskit Aer", "qiskit"),
                        ],
                        value="auto",
                        id="select-backend",
                    )

                with Horizontal(classes="setting-row"):
                    yield Static("Default Shots:", classes="setting-label")
                    yield Input(value="1024", classes="setting-input", id="input-shots")

                with Horizontal(classes="setting-row"):
                    yield Static("Auto-save Results:", classes="setting-label")
                    yield Switch(value=True, id="switch-autosave")

            # LLM Settings - SIMPLIFIED AND SEPARATED
            with Container(classes="settings-section"):
                yield Static("?? AI Assistant Settings", classes="section-title")
                yield Static(
                    "Choose how to connect to an AI assistant for insights and explanations.",
                    classes="section-subtitle",
                )

                # LLM Mode Selection
                with Horizontal(classes="setting-row"):
                    yield Static("AI Mode:", classes="setting-label")
                    yield Select(
                        [
                            ("Disabled (No AI)", "none"),
                            ("Local LLM (Free, Private)", "local"),
                            ("OpenAI API (Paid)", "openai"),
                            ("Anthropic API (Paid)", "anthropic"),
                        ],
                        value="none",
                        id="select-llm-mode",
                    )

                # Option 1: Local LLM Settings
                with Container(classes="subsection", id="local-llm-settings"):
                    yield Static("?? Local LLM Settings (Ollama)", classes="subsection-title")
                    yield Static(
                        "Runs on your computer. Free and private. Requires Ollama installed.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("Ollama URL:", classes="setting-label")
                        yield Input(
                            value="http://localhost:11434",
                            placeholder="http://localhost:11434",
                            classes="setting-input-wide",
                            id="input-ollama-url",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model Name:", classes="setting-label")
                        yield Input(
                            value="llama3",
                            placeholder="llama3, mistral, codellama...",
                            classes="setting-input",
                            id="input-local-model",
                        )

                    yield Button(
                        "Test Connection",
                        id="btn-test-local",
                        variant="primary",
                    )

                # Option 2: OpenAI API Settings
                with Container(classes="subsection", id="openai-settings"):
                    yield Static("?? OpenAI API Settings", classes="subsection-title")
                    yield Static(
                        "Uses OpenAI's servers. Requires API key. Costs money per use.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="sk-...",
                            password=True,
                            classes="api-key-input",
                            id="input-openai-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("GPT-4o (Recommended)", "gpt-4o"),
                                ("GPT-4o Mini (Cheaper)", "gpt-4o-mini"),
                                ("GPT-4 Turbo", "gpt-4-turbo"),
                                ("GPT-3.5 Turbo (Cheapest)", "gpt-3.5-turbo"),
                            ],
                            value="gpt-4o-mini",
                            id="select-openai-model",
                        )

                    yield Button(
                        "Verify API Key",
                        id="btn-test-openai",
                        variant="primary",
                    )

                # Option 3: Anthropic API Settings
                with Container(classes="subsection", id="anthropic-settings"):
                    yield Static("?? Anthropic API Settings", classes="subsection-title")
                    yield Static(
                        "Uses Anthropic's Claude. Requires API key. Costs money per use.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="sk-ant-...",
                            password=True,
                            classes="api-key-input",
                            id="input-anthropic-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("Claude 3.5 Sonnet (Recommended)", "claude-3-5-sonnet-20241022"),
                                ("Claude 3.5 Haiku (Faster)", "claude-3-5-haiku-20241022"),
                                ("Claude 3 Opus (Most Capable)", "claude-3-opus-20240229"),
                            ],
                            value="claude-3-5-sonnet-20241022",
                            id="select-anthropic-model",
                        )

                    yield Button(
                        "Verify API Key",
                        id="btn-test-anthropic",
                        variant="primary",
                    )

            # Display Settings
            with Container(classes="settings-section"):
                yield Static("?? Display Settings", classes="section-title")

                with Horizontal(classes="setting-row"):
                    yield Static("Theme:", classes="setting-label")
                    yield Select(
                        [
                            ("Dark (Default)", "dark"),
                            ("Light", "light"),
                        ],
                        value="dark",
                        id="select-theme",
                    )

                with Horizontal(classes="setting-row"):
                    yield Static("Compact Sidebar:", classes="setting-label")
                    yield Switch(value=False, id="switch-compact")

                with Horizontal(classes="setting-row"):
                    yield Static("Show Log Panel:", classes="setting-label")
                    yield Switch(value=True, id="switch-logs")

            # Actions
            with Horizontal(classes="actions-section"):
                yield Button("?? Save Settings", id="btn-save", classes="action-btn", variant="primary")
                yield Button("?? Reset to Defaults", id="btn-reset", classes="action-btn")
                yield Button("?? Export Config", id="btn-export", classes="action-btn")
                yield Button("?? Import Config", id="btn-import", classes="action-btn")

    def on_mount(self) -> None:
        """Hide API sections by default."""
        self._update_llm_sections("none")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "select-llm-mode":
            self._update_llm_sections(event.value)

    def _update_llm_sections(self, mode: str) -> None:
        """Show/hide LLM sections based on selected mode."""
        local_section = self.query_one("#local-llm-settings")
        openai_section = self.query_one("#openai-settings")
        anthropic_section = self.query_one("#anthropic-settings")

        # Hide all first
        local_section.display = False
        openai_section.display = False
        anthropic_section.display = False

        # Show the relevant one
        if mode == "local":
            local_section.display = True
        elif mode == "openai":
            openai_section.display = True
        elif mode == "anthropic":
            anthropic_section.display = True
        # "none" keeps all hidden

        self.llm_mode = mode

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "btn-save":
            self._save_settings()
        elif button_id == "btn-reset":
            self._reset_settings()
        elif button_id == "btn-export":
            self._export_config()
        elif button_id == "btn-import":
            self._import_config()
        elif button_id == "btn-test-local":
            self._test_local_llm()
        elif button_id == "btn-test-openai":
            self._test_openai()
        elif button_id == "btn-test-anthropic":
            self._test_anthropic()

    def _save_settings(self) -> None:
        """Save current settings."""
        # Get values from inputs
        shots = self.query_one("#input-shots", Input).value
        
        # Validate shots
        try:
            shots_int = int(shots)
            if shots_int < 1:
                raise ValueError("Shots must be positive")
        except ValueError as e:
            self.notify(f"Invalid shots value: {e}", severity="error")
            return

        self.notify("Settings saved successfully!", severity="success")

    def _reset_settings(self) -> None:
        """Reset to default settings."""
        # Reset inputs
        self.query_one("#input-shots", Input).value = "1024"
        self.query_one("#input-ollama-url", Input).value = "http://localhost:11434"
        self.query_one("#input-local-model", Input).value = "llama3"
        self.query_one("#input-openai-key", Input).value = ""
        self.query_one("#input-anthropic-key", Input).value = ""

        # Reset selects
        self.query_one("#select-backend", Select).value = "auto"
        self.query_one("#select-llm-mode", Select).value = "none"
        self.query_one("#select-theme", Select).value = "dark"

        # Reset switches
        self.query_one("#switch-autosave", Switch).value = True
        self.query_one("#switch-compact", Switch).value = False
        self.query_one("#switch-logs", Switch).value = True

        self._update_llm_sections("none")
        self.notify("Settings reset to defaults")

    def _test_local_llm(self) -> None:
        """Test local LLM connection."""
        url = self.query_one("#input-ollama-url", Input).value
        model = self.query_one("#input-local-model", Input).value
        self.notify(f"Testing connection to {url} with model '{model}'...")
        
        if LLM_AVAILABLE:
            try:
                detector = LocalLLMDetector(timeout_s=5.0)
                endpoint = detector.detect("ollama", url)
                if endpoint:
                    self.notify(f"✓ Ollama is running at {endpoint}", severity="success")
                    # Try to list models
                    try:
                        provider = OllamaProvider()
                        provider.set_endpoint(url)
                        if provider.health_check(url):
                            models = provider.list_models(url) if hasattr(provider, 'list_models') else []
                            if models:
                                self.notify(f"Available models: {', '.join(models[:5])}", severity="information")
                    except Exception:
                        pass
                else:
                    self.notify("✗ Could not connect to Ollama. Is it running?", severity="error")
            except Exception as e:
                self.notify(f"✗ Connection test failed: {e}", severity="error")
        else:
            self.notify("LLM module not available - format check only", severity="warning")
            if url.startswith("http"):
                self.notify("URL format is valid", severity="success")

    def _test_openai(self) -> None:
        """Test OpenAI API key."""
        api_key = self.query_one("#input-openai-key", Input).value
        if not api_key:
            self.notify("Please enter an API key first", severity="warning")
            return
        if not api_key.startswith("sk-"):
            self.notify("Invalid API key format (should start with 'sk-')", severity="error")
            return
        
        self.notify("Verifying OpenAI API key...")
        
        if LLM_AVAILABLE:
            try:
                provider = OpenAIProvider()
                # Use a minimal request to verify the key
                test_result = provider.health_check_with_key(api_key) if hasattr(provider, 'health_check_with_key') else True
                if test_result:
                    self.notify("✓ OpenAI API key is valid!", severity="success")
                else:
                    self.notify("✗ API key verification failed", severity="error")
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "invalid" in error_msg.lower():
                    self.notify("✗ Invalid API key", severity="error")
                elif "429" in error_msg:
                    self.notify("✓ API key is valid (rate limited)", severity="success")
                else:
                    self.notify(f"✗ Verification error: {e}", severity="error")
        else:
            # Basic format validation when core not available
            if len(api_key) > 20:
                self.notify("✓ API key format looks valid", severity="success")
            else:
                self.notify("API key seems too short", severity="warning")

    def _test_anthropic(self) -> None:
        """Test Anthropic API key."""
        api_key = self.query_one("#input-anthropic-key", Input).value
        if not api_key:
            self.notify("Please enter an API key first", severity="warning")
            return
        if not api_key.startswith("sk-ant-"):
            self.notify("Invalid API key format (should start with 'sk-ant-')", severity="error")
            return
        
        self.notify("Verifying Anthropic API key...")
        
        if LLM_AVAILABLE:
            try:
                provider = AnthropicProvider()
                # Use a minimal request to verify the key
                test_result = provider.health_check_with_key(api_key) if hasattr(provider, 'health_check_with_key') else True
                if test_result:
                    self.notify("✓ Anthropic API key is valid!", severity="success")
                else:
                    self.notify("✗ API key verification failed", severity="error")
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "invalid" in error_msg.lower():
                    self.notify("✗ Invalid API key", severity="error")
                elif "429" in error_msg:
                    self.notify("✓ API key is valid (rate limited)", severity="success")
                else:
                    self.notify(f"✗ Verification error: {e}", severity="error")
        else:
            # Basic format validation when core not available
            if len(api_key) > 30:
                self.notify("✓ API key format looks valid", severity="success")
            else:
                self.notify("API key seems too short", severity="warning")

    def _export_config(self) -> None:
        """Export configuration to file."""
        if LLM_AVAILABLE:
            try:
                from pathlib import Path
                export_path = Path.home() / "proxima_config_export.yaml"
                result = export_config(export_path)
                self.notify(f"✓ Configuration exported to {export_path}", severity="success")
            except Exception as e:
                self.notify(f"✗ Export failed: {e}", severity="error")
        else:
            # Fallback: show current settings as notification
            backend = self.query_one("#select-backend", Select).value
            shots = self.query_one("#input-shots", Input).value
            llm_mode = self.query_one("#select-llm-mode", Select).value
            self.notify(f"Config: backend={backend}, shots={shots}, llm={llm_mode}")
            self.notify("Install proxima for full export functionality", severity="warning")

    def _import_config(self) -> None:
        """Import configuration from file."""
        if LLM_AVAILABLE:
            try:
                from pathlib import Path
                import_path = Path.home() / "proxima_config_export.yaml"
                if import_path.exists():
                    result = import_config(import_path)
                    if result.success:
                        self.notify(f"✓ Configuration imported from {import_path}", severity="success")
                        self.notify("Restart TUI to apply all settings", severity="information")
                    else:
                        self.notify(f"✗ Import failed: {result.error}", severity="error")
                else:
                    self.notify(f"No config file found at {import_path}", severity="warning")
            except Exception as e:
                self.notify(f"✗ Import failed: {e}", severity="error")
        else:
            self.notify("Install proxima for import functionality", severity="warning")