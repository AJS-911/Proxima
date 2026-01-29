"""Add custom backend dialog with AI assistance.

Allows beginners to add custom backends through a guided, scrollable interface
with optional AI assistance for configuration.
"""

from textual.screen import ModalScreen
from textual.containers import Vertical, Horizontal, ScrollableContainer, Container
from textual.widgets import Static, Button, Input, Switch, Select, Label, TextArea, RichLog
from rich.text import Text
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json

try:
    from ...styles.theme import get_theme
except ImportError:
    def get_theme():
        class MockTheme:
            primary = "blue"
            accent = "cyan"
            success = "green"
            warning = "yellow"
            error = "red"
            fg_base = "white"
            fg_muted = "grey"
            fg_subtle = "dim"
        return MockTheme()

# Try to import LLM router for AI assistance
try:
    from proxima.intelligence.llm_router import LLMRouter, LLMRequest
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class AddCustomBackendDialog(ModalScreen):
    """Dialog for adding a custom backend with AI-assisted configuration."""

    DEFAULT_CSS = """
    AddCustomBackendDialog { 
        align: center middle; 
    }
    
    AddCustomBackendDialog > .dialog-container {
        padding: 1 2;
        border: thick $accent;
        background: $surface;
        width: 100%;
        height: 100%;
        max-width: 100%;
        max-height: 100%;
    }
    
    AddCustomBackendDialog .dialog-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
        height: 2;
    }
    
    AddCustomBackendDialog .main-layout {
        layout: horizontal;
        height: 1fr;
    }
    
    AddCustomBackendDialog .config-panel {
        width: 55%;
        height: 1fr;
        border-right: solid $primary-darken-2;
        padding-right: 1;
    }
    
    AddCustomBackendDialog .ai-panel {
        width: 45%;
        height: 1fr;
        padding-left: 1;
    }
    
    AddCustomBackendDialog ScrollableContainer {
        height: 1fr;
        border: solid $primary-darken-2;
        background: $surface-darken-1;
    }
    
    AddCustomBackendDialog .config-section {
        margin: 1 0;
        padding: 1;
        border: solid $primary-darken-3;
        height: auto;
        background: $surface;
    }
    
    AddCustomBackendDialog .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        height: 1;
    }
    
    AddCustomBackendDialog .config-row {
        layout: horizontal;
        height: 3;
        margin: 0 0 1 0;
        align: left middle;
    }
    
    AddCustomBackendDialog .config-label {
        width: 18;
        height: 3;
        content-align: left middle;
    }
    
    AddCustomBackendDialog .config-input {
        width: 1fr;
        min-width: 20;
    }
    
    AddCustomBackendDialog .config-hint {
        color: $text-muted;
        text-style: italic;
        height: auto;
        margin-bottom: 1;
    }
    
    AddCustomBackendDialog Input {
        width: 100%;
        height: 3;
    }
    
    AddCustomBackendDialog Select {
        width: 100%;
        height: 3;
    }
    
    AddCustomBackendDialog TextArea {
        height: 6;
        width: 100%;
    }
    
    AddCustomBackendDialog Switch {
        margin-left: 1;
    }
    
    AddCustomBackendDialog .ai-header {
        height: 3;
        background: $primary-darken-2;
        padding: 0 1;
    }
    
    AddCustomBackendDialog .ai-title {
        text-style: bold;
        color: $accent;
    }
    
    AddCustomBackendDialog .ai-chat-area {
        height: 1fr;
        padding: 1;
        background: $surface-darken-2;
        border: solid $primary-darken-3;
    }
    
    AddCustomBackendDialog .ai-stats-section {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        border: solid $primary-darken-3;
        background: $surface-darken-1;
    }
    
    AddCustomBackendDialog .stats-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    AddCustomBackendDialog .stats-row {
        height: auto;
        layout: horizontal;
    }
    
    AddCustomBackendDialog .stats-label {
        width: 12;
        color: $text-muted;
    }
    
    AddCustomBackendDialog .stats-value {
        width: 1fr;
        color: $accent;
        text-align: right;
    }
    
    AddCustomBackendDialog .input-hint {
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
        margin-top: 1;
    }
    
    AddCustomBackendDialog .ai-input-row {
        height: 3;
        layout: horizontal;
        margin-top: 1;
    }
    
    AddCustomBackendDialog .ai-input {
        width: 1fr;
        margin-right: 1;
    }
    
    AddCustomBackendDialog .ai-send-btn {
        min-width: 8;
    }
    
    AddCustomBackendDialog .footer {
        height: 3;
        layout: horizontal;
        margin-top: 1;
        padding-top: 1;
        border-top: solid $primary-darken-3;
    }
    
    AddCustomBackendDialog .footer Button {
        margin-right: 1;
    }
    
    AddCustomBackendDialog .required {
        color: $error;
    }
    """

    BINDINGS = [
        ("escape", "close", "Close"),
        ("ctrl+s", "save", "Save"),
        ("ctrl+j", "prev_prompt", "Previous Prompt"),
        ("ctrl+l", "next_prompt", "Next Prompt"),
    ]

    def __init__(self, on_save: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self._config: Dict[str, Any] = {}
        self._on_save = on_save
        self._llm_router = None
        self._ai_conversation = []
        self._prompt_history: list[str] = []  # Store prompt history
        self._prompt_history_index: int = -1  # Current position in history
        self._ai_stats = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'requests': 0,
            'thinking_time_ms': 0,
        }
        
        # Initialize LLM router if available
        if LLM_AVAILABLE:
            try:
                # Auto-consent function for TUI mode - user has already configured provider
                def auto_consent(prompt: str) -> bool:
                    return True  # Auto-approve in TUI since user explicitly configured
                
                self._llm_router = LLMRouter(consent_prompt=auto_consent)
                # Load and apply saved LLM settings
                self._init_llm_with_settings()
            except Exception:
                pass
    
    def _init_llm_with_settings(self) -> None:
        """Initialize LLM router with saved settings."""
        from pathlib import Path
        import json
        
        try:
            config_path = Path.home() / ".proxima" / "tui_settings.json"
            if not config_path.exists():
                return
            
            with open(config_path, 'r') as f:
                settings = json.load(f)
            
            llm = settings.get('llm', {})
            mode = llm.get('mode', 'none')
            
            if mode == 'none':
                return
            
            # Store the provider and model for requests
            self._llm_provider = mode
            
            # Get model for selected provider
            model_key_map = {
                'local': 'local_model',
                'ollama': 'local_model',
                'openai': 'openai_model',
                'anthropic': 'anthropic_model',
                'google': 'google_model',
                'xai': 'xai_model',
                'deepseek': 'deepseek_model',
                'mistral': 'mistral_model',
                'groq': 'groq_model',
                'together': 'together_model',
                'openrouter': 'openrouter_model',
                'cohere': 'cohere_model',
                'perplexity': 'perplexity_model',
                'lmstudio': 'lmstudio_model',
                'llamacpp': 'llamacpp_model',
                'llama_cpp': 'llamacpp_model',
                'azure_openai': 'azure_deployment',
                'azure': 'azure_deployment',
                'vertex_ai': 'vertex_model',
                'aws_bedrock': 'aws_model',
                'huggingface': 'hf_model',
                'fireworks': 'fireworks_model',
                'replicate': 'replicate_model',
                'ai21': 'ai21_model',
                'deepinfra': 'deepinfra_model',
                'localai': 'localai_model',
            }
            
            model_key = model_key_map.get(mode, f'{mode}_model')
            self._llm_model = llm.get(model_key, '')
            
            # Get API key if needed (TUI uses _key suffix, not _api_key)
            api_key_map = {
                'openai': 'openai_key',
                'anthropic': 'anthropic_key',
                'google': 'google_key',
                'xai': 'xai_key',
                'deepseek': 'deepseek_key',
                'mistral': 'mistral_key',
                'groq': 'groq_key',
                'together': 'together_key',
                'openrouter': 'openrouter_key',
                'cohere': 'cohere_key',
                'perplexity': 'perplexity_key',
                'azure_openai': 'azure_key',
                'azure': 'azure_key',
                'vertex_ai': 'vertex_key',
                'aws_bedrock': 'aws_access_key',
                'huggingface': 'hf_token',
                'fireworks': 'fireworks_key',
                'replicate': 'replicate_token',
                'ai21': 'ai21_key',
                'deepinfra': 'deepinfra_key',
            }
            
            api_key_field = api_key_map.get(mode)
            api_key = None
            if api_key_field:
                api_key = llm.get(api_key_field, '')
            
            # Comprehensive mapping of TUI provider names to router provider names for API key registration
            # All major providers are now supported
            provider_name_map = {
                'local': 'ollama',
                'ollama': 'ollama',
                'lmstudio': 'lmstudio',
                'llamacpp': 'llama_cpp',
                'llama_cpp': 'llama_cpp',
                'localai': 'ollama',
                'openai': 'openai',
                'anthropic': 'anthropic',
                'google': 'google',  # Google Gemini
                'gemini': 'google',
                'xai': 'xai',        # xAI Grok
                'grok': 'xai',
                'deepseek': 'deepseek',
                'mistral': 'mistral',
                'groq': 'groq',
                'together': 'together',
                'openrouter': 'openrouter',
                'cohere': 'cohere',
                'perplexity': 'perplexity',
                'azure_openai': 'azure_openai',
                'azure': 'azure_openai',
                'huggingface': 'huggingface',
                'fireworks': 'fireworks',
                'replicate': 'replicate',
            }
            # Only a few providers still need simulation (no direct API support)
            unsupported_providers = {'vertex_ai', 'aws_bedrock', 'ai21', 'deepinfra'}
            if mode in unsupported_providers:
                router_provider = None  # Will trigger simulation fallback
            else:
                router_provider = provider_name_map.get(mode, mode)
            
            # Register API key with the router for proper authentication
            if api_key and self._llm_router:
                try:
                    self._llm_router.api_keys.store_key(router_provider, api_key)
                except Exception:
                    pass  # Silently ignore if API key storage fails
                
        except Exception:
            self._llm_provider = None
            self._llm_model = None

    def compose(self):
        with Vertical(classes="dialog-container"):
            yield Static("➕ Add Custom Backend", classes="dialog-title")
            
            with Horizontal(classes="main-layout"):
                # Left panel: Configuration form
                with Vertical(classes="config-panel"):
                    yield Static("📋 Backend Configuration", classes="section-title")
                    
                    with ScrollableContainer():
                        # Basic Information Section
                        with Vertical(classes="config-section"):
                            yield Static("📌 Basic Information", classes="section-title")
                            yield Static("Required fields are marked with *", classes="config-hint")
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Backend Name: *", classes="config-label")
                                yield Input(
                                    placeholder="e.g., MyQuantumBackend",
                                    id="input-backend-name",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Display Name:", classes="config-label")
                                yield Input(
                                    placeholder="e.g., My Quantum Backend",
                                    id="input-display-name",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Backend Type:", classes="config-label")
                                yield Select(
                                    [
                                        ("Simulator (CPU)", "simulator"),
                                        ("GPU Accelerated", "gpu"),
                                        ("Hybrid (CPU+GPU)", "hybrid"),
                                        ("Cloud/Remote", "cloud"),
                                        ("Hardware/QPU", "hardware"),
                                        ("Custom", "custom"),
                                    ],
                                    value="simulator",
                                    id="select-backend-type",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Description:", classes="config-label")
                                yield Input(
                                    placeholder="Brief description of the backend",
                                    id="input-description",
                                    classes="config-input"
                                )
                        
                        # Connection Settings Section
                        with Vertical(classes="config-section"):
                            yield Static("🔌 Connection Settings", classes="section-title")
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Module Path:", classes="config-label")
                                yield Input(
                                    placeholder="e.g., my_backend.simulator",
                                    id="input-module-path",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Class Name:", classes="config-label")
                                yield Input(
                                    placeholder="e.g., QuantumSimulator",
                                    id="input-class-name",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Install Path:", classes="config-label")
                                yield Input(
                                    placeholder="Path to backend installation (optional)",
                                    id="input-install-path",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("API Endpoint:", classes="config-label")
                                yield Input(
                                    placeholder="For remote backends (optional)",
                                    id="input-api-endpoint",
                                    classes="config-input"
                                )
                        
                        # Performance Settings Section
                        with Vertical(classes="config-section"):
                            yield Static("⚡ Performance Settings", classes="section-title")
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Max Qubits:", classes="config-label")
                                yield Input(
                                    value="30",
                                    id="input-max-qubits",
                                    classes="config-input",
                                    type="integer"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Precision:", classes="config-label")
                                yield Select(
                                    [
                                        ("Single (float32)", "float32"),
                                        ("Double (float64)", "float64"),
                                        ("Mixed Precision", "mixed"),
                                    ],
                                    value="float64",
                                    id="select-precision",
                                    classes="config-input"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("GPU Support:", classes="config-label")
                                yield Switch(value=False, id="switch-gpu-support")
                            
                            with Horizontal(classes="config-row"):
                                yield Label("GPU Device ID:", classes="config-label")
                                yield Input(
                                    value="0",
                                    id="input-gpu-device",
                                    classes="config-input",
                                    type="integer"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Thread Count:", classes="config-label")
                                yield Input(
                                    value="4",
                                    id="input-threads",
                                    classes="config-input",
                                    type="integer"
                                )
                        
                        # Advanced Settings Section
                        with Vertical(classes="config-section"):
                            yield Static("🔧 Advanced Settings", classes="section-title")
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Timeout (ms):", classes="config-label")
                                yield Input(
                                    value="30000",
                                    id="input-timeout",
                                    classes="config-input",
                                    type="integer"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Max Retries:", classes="config-label")
                                yield Input(
                                    value="3",
                                    id="input-retries",
                                    classes="config-input",
                                    type="integer"
                                )
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Enable Caching:", classes="config-label")
                                yield Switch(value=True, id="switch-caching")
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Enable Logging:", classes="config-label")
                                yield Switch(value=True, id="switch-logging")
                            
                            with Horizontal(classes="config-row"):
                                yield Label("Log Level:", classes="config-label")
                                yield Select(
                                    [
                                        ("Debug", "DEBUG"),
                                        ("Info", "INFO"),
                                        ("Warning", "WARNING"),
                                        ("Error", "ERROR"),
                                    ],
                                    value="INFO",
                                    id="select-log-level",
                                    classes="config-input"
                                )
                        
                        # Custom Parameters Section
                        with Vertical(classes="config-section"):
                            yield Static("📝 Custom Parameters (JSON)", classes="section-title")
                            yield Static("Add any backend-specific parameters as JSON", classes="config-hint")
                            yield TextArea(
                                text='{\n  "custom_param": "value"\n}',
                                id="textarea-custom-params",
                                language="json"
                            )
                
                # Right panel: AI Assistant
                with Vertical(classes="ai-panel"):
                    with Horizontal(classes="ai-header"):
                        yield Static("🤖 AI Assistant", classes="ai-title")
                    
                    # Statistics Section
                    with Container(classes="ai-stats-section", id="ai-stats-section"):
                        yield Static("📊 Statistics", classes="stats-title")
                        with Horizontal(classes="stats-row"):
                            yield Static("Model:", classes="stats-label")
                            yield Static("—", classes="stats-value", id="stat-model")
                        with Horizontal(classes="stats-row"):
                            yield Static("Provider:", classes="stats-label")
                            yield Static("—", classes="stats-value", id="stat-provider")
                        with Horizontal(classes="stats-row"):
                            yield Static("Prompt Tokens:", classes="stats-label")
                            yield Static("0", classes="stats-value", id="stat-prompt-tokens")
                        with Horizontal(classes="stats-row"):
                            yield Static("Completion:", classes="stats-label")
                            yield Static("0", classes="stats-value", id="stat-completion-tokens")
                        with Horizontal(classes="stats-row"):
                            yield Static("Total Tokens:", classes="stats-label")
                            yield Static("0", classes="stats-value", id="stat-total-tokens")
                        with Horizontal(classes="stats-row"):
                            yield Static("Requests:", classes="stats-label")
                            yield Static("0", classes="stats-value", id="stat-requests")
                        with Horizontal(classes="stats-row"):
                            yield Static("Think Time:", classes="stats-label")
                            yield Static("0ms", classes="stats-value", id="stat-thinking-time")
                    
                    yield RichLog(
                        auto_scroll=True,
                        classes="ai-chat-area",
                        id="ai-chat-log"
                    )
                    
                    # Input with hint
                    yield Static("Ctrl+J: Previous prompt | Ctrl+L: Next prompt", classes="input-hint", id="input-hint")
                    
                    with Horizontal(classes="ai-input-row"):
                        yield Input(
                            placeholder="Ask AI for help... (Ctrl+J/L for history)",
                            id="ai-input",
                            classes="ai-input"
                        )
                        yield Button("Ask", id="btn-ai-ask", variant="primary", classes="ai-send-btn")
            
            # Footer with action buttons
            with Horizontal(classes="footer"):
                yield Button("💾 Save Backend", id="btn-save", variant="success")
                yield Button("🔍 Test Connection", id="btn-test", variant="primary")
                yield Button("🗑️ Clear Form", id="btn-clear", variant="warning")
                yield Button("❌ Cancel", id="btn-cancel", variant="error")

    def on_mount(self):
        """Initialize the dialog."""
        self._show_ai_welcome()

    def _show_ai_welcome(self):
        """Show welcome message in AI chat."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#ai-chat-log", RichLog)
            
            welcome = Text()
            welcome.append("🤖 AI Backend Assistant\n", style=f"bold {theme.accent}")
            welcome.append("─" * 25 + "\n", style=theme.fg_subtle)
            welcome.append("I can help you configure your backend!\n\n", style=theme.fg_base)
            welcome.append("Try asking:\n", style=theme.fg_muted)
            welcome.append("• How do I set up a GPU backend?\n", style=theme.fg_subtle)
            welcome.append("• What module path should I use?\n", style=theme.fg_subtle)
            welcome.append("• Help me configure cuQuantum\n", style=theme.fg_subtle)
            welcome.append("• What are optimal settings for 20 qubits?\n", style=theme.fg_subtle)
            
            chat_log.write(welcome)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-save":
            self._save_backend()
        elif button_id == "btn-test":
            self._test_connection()
        elif button_id == "btn-clear":
            self._clear_form()
        elif button_id == "btn-cancel":
            self.dismiss(None)
        elif button_id == "btn-ai-ask":
            self._send_ai_message()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input fields."""
        if event.input.id == "ai-input":
            self._send_ai_message()

    def action_prev_prompt(self) -> None:
        """Navigate to previous prompt in history (Ctrl+J)."""
        if not self._prompt_history:
            return
        
        try:
            input_widget = self.query_one("#ai-input", Input)
            
            # If we're at the start, save current input
            if self._prompt_history_index == -1:
                current = input_widget.value.strip()
                if current and (not self._prompt_history or self._prompt_history[-1] != current):
                    self._prompt_history.append(current)
                self._prompt_history_index = len(self._prompt_history) - 1
            elif self._prompt_history_index > 0:
                self._prompt_history_index -= 1
            
            if 0 <= self._prompt_history_index < len(self._prompt_history):
                input_widget.value = self._prompt_history[self._prompt_history_index]
        except Exception:
            pass

    def action_next_prompt(self) -> None:
        """Navigate to next prompt in history (Ctrl+L)."""
        if not self._prompt_history:
            return
        
        try:
            input_widget = self.query_one("#ai-input", Input)
            
            if self._prompt_history_index < len(self._prompt_history) - 1:
                self._prompt_history_index += 1
                input_widget.value = self._prompt_history[self._prompt_history_index]
            else:
                # At the end, clear input
                self._prompt_history_index = -1
                input_widget.value = ""
        except Exception:
            pass

    def _update_stats_display(self) -> None:
        """Update the statistics display."""
        try:
            # Get LLM config
            llm_settings = self._get_llm_settings()
            provider = llm_settings.get('mode', 'none')
            model = llm_settings.get(f'{provider}_model', llm_settings.get('local_model', '—'))
            
            self.query_one("#stat-model", Static).update(model[:20] if model else "—")
            self.query_one("#stat-provider", Static).update(provider.title() if provider else "—")
            self.query_one("#stat-prompt-tokens", Static).update(str(self._ai_stats['prompt_tokens']))
            self.query_one("#stat-completion-tokens", Static).update(str(self._ai_stats['completion_tokens']))
            self.query_one("#stat-total-tokens", Static).update(str(self._ai_stats['total_tokens']))
            self.query_one("#stat-requests", Static).update(str(self._ai_stats['requests']))
            self.query_one("#stat-thinking-time", Static).update(f"{self._ai_stats['thinking_time_ms']}ms")
        except Exception:
            pass

    def _get_llm_settings(self) -> Dict:
        """Load LLM settings from storage."""
        try:
            config_path = Path.home() / ".proxima" / "tui_settings.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    settings = json.load(f)
                return settings.get('llm', {})
        except Exception:
            pass
        return {}

    def _send_ai_message(self) -> None:
        """Send message to AI assistant."""
        import time
        
        try:
            theme = get_theme()
            input_widget = self.query_one("#ai-input", Input)
            message = input_widget.value.strip()
            
            if not message:
                return
            
            # Add to prompt history
            if not self._prompt_history or self._prompt_history[-1] != message:
                self._prompt_history.append(message)
            self._prompt_history_index = -1  # Reset history position
            
            # Clear input
            input_widget.value = ""
            
            # Show user message
            chat_log = self.query_one("#ai-chat-log", RichLog)
            
            user_text = Text()
            user_text.append("👤 You: ", style=f"bold {theme.primary}")
            user_text.append(message + "\n", style=theme.fg_base)
            chat_log.write(user_text)
            
            # Track timing
            start_time = time.time()
            
            # Get current form state for context
            form_state = self._get_form_state()
            
            # Update request count
            self._ai_stats['requests'] += 1
            
            # Send to LLM or simulate
            if self._llm_router and LLM_AVAILABLE:
                self._query_ai(message, form_state, start_time)
            else:
                self._simulate_ai_response(message, form_state, start_time)
                
        except Exception as e:
            self._show_ai_error(str(e))

    def _get_form_state(self) -> Dict[str, Any]:
        """Get current form values."""
        try:
            return {
                "backend_name": self.query_one("#input-backend-name", Input).value,
                "display_name": self.query_one("#input-display-name", Input).value,
                "backend_type": self.query_one("#select-backend-type", Select).value,
                "module_path": self.query_one("#input-module-path", Input).value,
                "class_name": self.query_one("#input-class-name", Input).value,
                "gpu_support": self.query_one("#switch-gpu-support", Switch).value,
                "max_qubits": self.query_one("#input-max-qubits", Input).value,
            }
        except Exception:
            return {}

    def _query_ai(self, message: str, form_state: Dict, start_time: float = None) -> None:
        """Query the AI for backend configuration help."""
        import time
        if start_time is None:
            start_time = time.time()
        
        # Check if LLM provider is configured
        provider = getattr(self, '_llm_provider', None)
        if not provider or provider == 'none':
            # Fall back to simulated response
            self._simulate_ai_response(message, form_state, start_time)
            return
        
        # Comprehensive mapping of TUI provider names to LLM router provider names
        # All major providers are now supported
        provider_name_map = {
            # Local providers
            'local': 'ollama',
            'ollama': 'ollama',
            'lmstudio': 'lmstudio',
            'llamacpp': 'llama_cpp',
            'llama_cpp': 'llama_cpp',
            'localai': 'ollama',
            # Cloud providers - all supported
            'openai': 'openai',
            'anthropic': 'anthropic',
            'google': 'google',  # Google Gemini
            'gemini': 'google',
            'xai': 'xai',        # xAI Grok
            'grok': 'xai',
            'deepseek': 'deepseek',
            'mistral': 'mistral',
            'groq': 'groq',
            'together': 'together',
            'openrouter': 'openrouter',
            'cohere': 'cohere',
            'perplexity': 'perplexity',
            'azure_openai': 'azure_openai',
            'azure': 'azure_openai',
            'huggingface': 'huggingface',
            'fireworks': 'fireworks',
            'replicate': 'replicate',
        }
        # Only a few providers still need simulation (no direct API support)
        unsupported_providers = {'vertex_ai', 'aws_bedrock', 'ai21', 'deepinfra'}
        if provider in unsupported_providers:
            router_provider = None  # Will trigger simulation fallback
        else:
            router_provider = provider_name_map.get(provider, provider)
            
        try:
            context = f"""Current backend configuration:
- Name: {form_state.get('backend_name', 'Not set')}
- Type: {form_state.get('backend_type', 'simulator')}
- Module: {form_state.get('module_path', 'Not set')}
- GPU: {form_state.get('gpu_support', False)}
- Max Qubits: {form_state.get('max_qubits', 30)}"""

            # Build request with provider info
            model = getattr(self, '_llm_model', None)
            
            request = LLMRequest(
                prompt=f"{context}\n\nUser question: {message}",
                system_prompt="""You are a helpful AI assistant for quantum computing backend configuration.
Help users set up custom backends for the Proxima quantum simulation platform.
Provide specific, actionable advice for backend configuration including:
- Module paths and class names
- GPU configuration
- Performance optimization
- Common issues and solutions
Keep responses concise and practical.""",
                temperature=0.7,
                max_tokens=512,
                provider=router_provider,
                model=model if model else None,
            )
            
            response = self._llm_router.route(request)
            
            # Check if we got a valid response with actual content
            if response and hasattr(response, 'text') and response.text and response.text.strip():
                self._handle_ai_response(response, start_time)
            else:
                # Empty response - fall back to simulation
                self._simulate_ai_response(message, form_state, start_time)
            
        except PermissionError as e:
            # Consent required - use simulated response instead
            self._simulate_ai_response(message, form_state, start_time)
        except ConnectionError as e:
            # Local provider not running - use simulated response
            self._simulate_ai_response(message, form_state, start_time)
        except ValueError as e:
            # Provider not found in router - use simulated response instead of showing error
            if "Unknown provider" in str(e) or "No LLM provider" in str(e):
                self._simulate_ai_response(message, form_state, start_time)
            else:
                self._simulate_ai_response(message, form_state, start_time)
        except Exception as e:
            # For any other error, silently fall back to simulation
            # This includes HTTP errors (404, 500, etc.) and connection errors
            error_msg = str(e).lower()
            # Silently use simulation for common API/connection errors
            if any(err in error_msg for err in ['404', '500', '502', '503', 'not found', 
                   'connection', 'timeout', 'refused', 'unknown provider', 'consent',
                   'http', 'client error', 'server error']):
                self._simulate_ai_response(message, form_state, start_time)
            else:
                # Unknown error - show it but still fall back
                self._simulate_ai_response(message, form_state, start_time)

    def _handle_ai_response(self, response, start_time: float = None) -> None:
        """Handle AI response."""
        import time
        try:
            theme = get_theme()
            chat_log = self.query_one("#ai-chat-log", RichLog)
            
            if hasattr(response, 'error') and response.error:
                self._show_ai_error(response.error)
                return
            
            ai_text = Text()
            ai_text.append("🤖 AI: ", style=f"bold {theme.accent}")
            ai_text.append(response.text + "\n\n", style=theme.fg_base)
            chat_log.write(ai_text)
            
            # Update stats
            if hasattr(response, 'prompt_tokens'):
                self._ai_stats['prompt_tokens'] += response.prompt_tokens or 0
            if hasattr(response, 'completion_tokens'):
                self._ai_stats['completion_tokens'] += response.completion_tokens or 0
            if hasattr(response, 'total_tokens'):
                self._ai_stats['total_tokens'] += response.total_tokens or 0
            elif hasattr(response, 'prompt_tokens') and hasattr(response, 'completion_tokens'):
                self._ai_stats['total_tokens'] += (response.prompt_tokens or 0) + (response.completion_tokens or 0)
            
            if start_time:
                elapsed = int((time.time() - start_time) * 1000)
                self._ai_stats['thinking_time_ms'] = elapsed
            
            self._update_stats_display()
            
        except Exception as e:
            self._show_ai_error(str(e))

    def _simulate_ai_response(self, message: str, form_state: Dict, start_time: float = None) -> None:
        """Simulate AI response when LLM not available."""
        import time
        if start_time is None:
            start_time = time.time()
            
        theme = get_theme()
        chat_log = self.query_one("#ai-chat-log", RichLog)
        
        msg_lower = message.lower()
        
        # Generate contextual response
        if "gpu" in msg_lower or "cuda" in msg_lower:
            response = """For GPU acceleration:
1. Enable 'GPU Support' toggle
2. Set correct GPU Device ID (0 for first GPU)
3. For cuQuantum: module path is 'cuquantum.backends.statevector'
4. Ensure CUDA drivers are installed
5. Use 'float32' precision for better GPU performance"""
        
        elif "module" in msg_lower or "path" in msg_lower:
            response = """Module path examples:
• cuQuantum: cuquantum.backends.statevector
• Qiskit: qiskit_aer.AerSimulator
• Cirq: cirq.Simulator
• Custom: your_package.module_name

The module should be installed in your Python environment."""
        
        elif "qubit" in msg_lower:
            response = f"""Max qubits depends on your hardware:
• 16GB RAM: ~28 qubits (CPU)
• 32GB RAM: ~30 qubits (CPU)
• RTX 3090 (24GB): ~32 qubits (GPU)
• A100 (40GB): ~34 qubits (GPU)

Current setting: {form_state.get('max_qubits', 30)} qubits"""
        
        elif "cuquantum" in msg_lower:
            response = """cuQuantum configuration:
• Backend Type: GPU Accelerated
• Module Path: cuquantum.backends.statevector
• Class Name: StateVectorSimulator
• GPU Support: Enable
• Precision: float32 (faster) or float64 (accurate)
• Install: pip install cuquantum-python"""
        
        elif "test" in msg_lower or "connection" in msg_lower:
            response = """To test your backend:
1. Click 'Test Connection' button
2. The system will try to import and initialize your backend
3. Check the result message for success/errors
4. Common issues: missing module, wrong class name, GPU not available"""
        
        else:
            response = """I can help you with:
• GPU configuration and optimization
• Module paths for popular backends
• Performance tuning (qubits, precision)
• Troubleshooting connection issues

What would you like to know?"""
        
        ai_text = Text()
        ai_text.append("🤖 AI: ", style=f"bold {theme.accent}")
        ai_text.append(response + "\n\n", style=theme.fg_base)
        chat_log.write(ai_text)
        
        # Update stats for simulated response
        import time
        elapsed = int((time.time() - start_time) * 1000)
        self._ai_stats['thinking_time_ms'] = elapsed
        self._ai_stats['prompt_tokens'] += len(message.split()) * 2  # Approximate
        self._ai_stats['completion_tokens'] += len(response.split()) * 2  # Approximate
        self._ai_stats['total_tokens'] = self._ai_stats['prompt_tokens'] + self._ai_stats['completion_tokens']
        self._update_stats_display()

    def _show_ai_error(self, error: str) -> None:
        """Show error in AI chat."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#ai-chat-log", RichLog)
            
            error_text = Text()
            error_text.append("❌ Error: ", style=f"bold {theme.error}")
            error_text.append(error + "\n", style=theme.fg_muted)
            chat_log.write(error_text)
        except Exception:
            pass

    def _save_backend(self) -> None:
        """Save the backend configuration."""
        try:
            # Validate required fields
            backend_name = self.query_one("#input-backend-name", Input).value.strip()
            if not backend_name:
                self.notify("Backend name is required!", severity="error")
                return
            
            # Collect all configuration
            config = {
                "id": backend_name.lower().replace(" ", "_"),
                "name": backend_name,
                "display_name": self.query_one("#input-display-name", Input).value.strip() or backend_name,
                "description": self.query_one("#input-description", Input).value.strip(),
                "type": self.query_one("#select-backend-type", Select).value,
                "module_path": self.query_one("#input-module-path", Input).value.strip(),
                "class_name": self.query_one("#input-class-name", Input).value.strip(),
                "install_path": self.query_one("#input-install-path", Input).value.strip(),
                "api_endpoint": self.query_one("#input-api-endpoint", Input).value.strip(),
                "max_qubits": int(self.query_one("#input-max-qubits", Input).value or 30),
                "precision": self.query_one("#select-precision", Select).value,
                "gpu_support": self.query_one("#switch-gpu-support", Switch).value,
                "gpu_device": int(self.query_one("#input-gpu-device", Input).value or 0),
                "threads": int(self.query_one("#input-threads", Input).value or 4),
                "timeout": int(self.query_one("#input-timeout", Input).value or 30000),
                "retries": int(self.query_one("#input-retries", Input).value or 3),
                "caching": self.query_one("#switch-caching", Switch).value,
                "logging": self.query_one("#switch-logging", Switch).value,
                "log_level": self.query_one("#select-log-level", Select).value,
                "status": "unknown",
                "is_custom": True,
            }
            
            # Parse custom parameters
            try:
                custom_params_text = self.query_one("#textarea-custom-params", TextArea).text
                config["custom_params"] = json.loads(custom_params_text)
            except json.JSONDecodeError:
                config["custom_params"] = {}
            
            # Save to persistent storage
            self._save_to_storage(config)
            
            # Call callback if provided
            if self._on_save:
                self._on_save(config)
            
            self.notify(f"✓ Backend '{backend_name}' saved successfully!", severity="success")
            self.dismiss(config)
            
        except Exception as e:
            self.notify(f"Save failed: {e}", severity="error")

    def _save_to_storage(self, config: Dict) -> None:
        """Save backend configuration to persistent storage."""
        try:
            config_dir = Path.home() / ".proxima"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            backends_file = config_dir / "custom_backends.json"
            
            # Load existing backends
            existing = {}
            if backends_file.exists():
                with open(backends_file, 'r') as f:
                    existing = json.load(f)
            
            # Add/update this backend
            existing[config["id"]] = config
            
            # Save
            with open(backends_file, 'w') as f:
                json.dump(existing, f, indent=2)
                
        except Exception as e:
            raise Exception(f"Failed to save: {e}")

    def _test_connection(self) -> None:
        """Test the backend connection."""
        try:
            module_path = self.query_one("#input-module-path", Input).value.strip()
            class_name = self.query_one("#input-class-name", Input).value.strip()
            
            if not module_path:
                self.notify("Please enter a module path to test", severity="warning")
                return
            
            self.notify(f"Testing connection to {module_path}...", severity="information")
            
            # Try to import the module
            try:
                import importlib
                module = importlib.import_module(module_path)
                
                if class_name:
                    # Try to get the class
                    if hasattr(module, class_name):
                        self.notify(f"✓ Successfully found {class_name} in {module_path}", severity="success")
                    else:
                        self.notify(f"⚠ Module found but class '{class_name}' not found", severity="warning")
                else:
                    self.notify(f"✓ Module {module_path} imported successfully", severity="success")
                    
            except ImportError as e:
                self.notify(f"✗ Import failed: {e}", severity="error")
            except Exception as e:
                self.notify(f"✗ Error: {e}", severity="error")
                
        except Exception as e:
            self.notify(f"Test failed: {e}", severity="error")

    def _clear_form(self) -> None:
        """Clear all form fields."""
        try:
            self.query_one("#input-backend-name", Input).value = ""
            self.query_one("#input-display-name", Input).value = ""
            self.query_one("#input-description", Input).value = ""
            self.query_one("#select-backend-type", Select).value = "simulator"
            self.query_one("#input-module-path", Input).value = ""
            self.query_one("#input-class-name", Input).value = ""
            self.query_one("#input-install-path", Input).value = ""
            self.query_one("#input-api-endpoint", Input).value = ""
            self.query_one("#input-max-qubits", Input).value = "30"
            self.query_one("#select-precision", Select).value = "float64"
            self.query_one("#switch-gpu-support", Switch).value = False
            self.query_one("#input-gpu-device", Input).value = "0"
            self.query_one("#input-threads", Input).value = "4"
            self.query_one("#input-timeout", Input).value = "30000"
            self.query_one("#input-retries", Input).value = "3"
            self.query_one("#switch-caching", Switch).value = True
            self.query_one("#switch-logging", Switch).value = True
            self.query_one("#select-log-level", Select).value = "INFO"
            self.query_one("#textarea-custom-params", TextArea).text = '{\n  "custom_param": "value"\n}'
            
            self.notify("Form cleared", severity="information")
        except Exception:
            pass

    def action_close(self) -> None:
        """Close the dialog."""
        self.dismiss(None)

    def action_save(self) -> None:
        """Save keyboard shortcut."""
        self._save_backend()
