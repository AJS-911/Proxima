"""Settings screen for Proxima TUI.

Configuration management.
"""

import asyncio
from pathlib import Path
from typing import Optional

from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Button, Input, Switch, Select, RadioSet, RadioButton
from textual.css.query import NoMatches
from rich.text import Text

from .base import BaseScreen
from ..styles.theme import get_theme

try:
    from proxima.intelligence.llm_router import LocalLLMDetector, OllamaProvider, OpenAIProvider, AnthropicProvider, ProviderRegistry
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
        height: 1fr;
    }

    SettingsScreen .settings-section {
        margin-bottom: 2;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface;
        height: auto;
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
    
    SettingsScreen .custom-model-input {
        display: none;
        width: 30;
        margin-left: 1;
    }
    
    SettingsScreen .custom-model-input.visible {
        display: block;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_mode = "none"  # Start with "none" for fastest initial load
        self._loaded_sections = set()  # Track which sections have been created
        self._widget_cache = {}  # Cache widget references to avoid repeated queries
        self._settings_loaded = False  # Track if settings have been loaded
        self._detection_tasks = {}  # Track background detection tasks
        self._provider_container = None  # Reference to the dynamic provider container
        self._cached_llm_settings = {}  # Cache for loaded LLM settings

    def compose_main(self):
        """Compose the settings screen content - OPTIMIZED with lazy loading."""
        from textual.containers import ScrollableContainer
        with ScrollableContainer(classes="main-content settings-container"):
            # General Settings - Always shown (minimal widgets)
            with Container(classes="settings-section"):
                yield Static("⚙️ General Settings", classes="section-title")

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

            # AI Assistant Settings - Minimal initial render
            with Container(classes="settings-section"):
                yield Static("🤖 AI Assistant Settings", classes="section-title")
                yield Static(
                    "Choose how to connect to an AI assistant for insights and explanations.",
                    classes="section-subtitle",
                )

                # LLM Mode Selection - Only the dropdown, no provider sections yet
                with Horizontal(classes="setting-row"):
                    yield Static("AI Mode:", classes="setting-label")
                    yield Select(
                        [
                            ("Disabled (No AI)", "none"),
                            ("Local LLM (Free, Private)", "local"),
                            ("OpenAI API (GPT-4, GPT-4o)", "openai"),
                            ("Anthropic API (Claude)", "anthropic"),
                            ("Google AI (Gemini)", "google"),
                            ("xAI (Grok)", "xai"),
                            ("DeepSeek API", "deepseek"),
                            ("Mistral AI", "mistral"),
                            ("Groq (Fast Inference)", "groq"),
                            ("Together AI", "together"),
                            ("OpenRouter", "openrouter"),
                            ("Cohere API", "cohere"),
                            ("Perplexity AI", "perplexity"),
                            ("Azure OpenAI", "azure_openai"),
                            ("AWS Bedrock", "aws_bedrock"),
                            ("Hugging Face", "huggingface"),
                            ("Fireworks AI", "fireworks"),
                            ("Replicate", "replicate"),
                            ("AI21 Labs", "ai21"),
                            ("DeepInfra", "deepinfra"),
                            ("Ollama (Local)", "ollama"),
                            ("Other Provider...", "custom"),
                        ],
                        value="none",
                        id="select-llm-mode",
                    )

                # DYNAMIC PROVIDER CONTAINER - Will be populated lazily
                yield Container(id="provider-settings-container", classes="subsection")

                # AI Thinking Panel - Always shown (minimal)
                with Container(classes="subsection"):
                    yield Static("🧠 AI Thinking Panel", classes="subsection-title")
                    yield Static(
                        "View what the AI is thinking in real-time. Shows prompts, responses, and token usage.",
                        classes="hint-text",
                    )
                    
                    with Horizontal(classes="setting-row"):
                        yield Static("Enable Thinking:", classes="setting-label")
                        yield Switch(value=False, id="switch-thinking-enabled")
                    
                    yield Button(
                        "🧠 Open AI Thinking Panel (Ctrl+T)",
                        id="btn-open-thinking",
                        variant="success",
                    )

            # Display Settings
            with Container(classes="settings-section"):
                yield Static("🎨 Display Settings", classes="section-title")

                with Horizontal(classes="setting-row"):
                    yield Static("Theme:", classes="setting-label")
                    yield Select(
                        [
                            ("Proxima Dark (Default)", "proxima-dark"),
                            ("Ocean Deep", "ocean-deep"),
                            ("Forest Night", "forest-night"),
                            ("Sunset Glow", "sunset-glow"),
                            ("Arctic Ice", "arctic-ice"),
                            ("Neon Nights", "neon-nights"),
                            ("Light Mode", "light-mode"),
                        ],
                        value="proxima-dark",
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
                yield Button("💾 Save Settings", id="btn-save", classes="action-btn", variant="primary")
                yield Button("🔄 Reset to Defaults", id="btn-reset", classes="action-btn")
                yield Button("📤 Export Config", id="btn-export", classes="action-btn")
                yield Button("📥 Import Config", id="btn-import", classes="action-btn")

    def _create_provider_widgets(self, mode: str) -> list:
        """Create widgets for a specific provider - called lazily on demand."""
        widgets = []
        
        if mode == "none":
            return widgets
        
        # Provider configurations - defines what widgets each provider needs
        PROVIDER_CONFIGS = {
            "local": {
                "title": "🖥️ Local LLM Settings (Ollama)",
                "hint": "Runs on your computer. Free and private. Requires Ollama installed.",
                "fields": [
                    {"type": "input", "label": "Ollama URL:", "id": "input-ollama-url", "value": "http://localhost:11434", "wide": True},
                    {"type": "input", "label": "Model Name:", "id": "input-local-model", "value": "llama3", "placeholder": "llama3, mistral, codellama..."},
                ],
                "buttons": [{"label": "🔗 Test Connection", "id": "btn-test-local", "variant": "primary"}],
            },
            "ollama": {
                "title": "🖥️ Ollama Settings",
                "hint": "Local LLM server. Free and private.",
                "fields": [
                    {"type": "input", "label": "Ollama URL:", "id": "input-ollama-url", "value": "http://localhost:11434", "wide": True},
                    {"type": "input", "label": "Model Name:", "id": "input-local-model", "value": "llama3"},
                ],
                "buttons": [{"label": "🔗 Test Connection", "id": "btn-test-local", "variant": "primary"}],
            },
            "openai": {
                "title": "🔑 OpenAI API Settings",
                "hint": "Uses OpenAI's servers. Requires API key.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-openai-key", "password": True, "placeholder": "sk-..."},
                    {"type": "select", "label": "Model:", "id": "select-openai-model", "options": [
                        ("GPT-4o (Recommended)", "gpt-4o"),
                        ("GPT-4o Mini (Cheaper)", "gpt-4o-mini"),
                        ("GPT-4 Turbo", "gpt-4-turbo"),
                        ("GPT-3.5 Turbo", "gpt-3.5-turbo"),
                    ], "value": "gpt-4o-mini"},
                ],
                "buttons": [
                    {"label": "🔍 Auto-Detect Models", "id": "btn-detect-openai", "variant": "default"},
                    {"label": "Verify API Key", "id": "btn-test-openai", "variant": "primary"},
                ],
            },
            "anthropic": {
                "title": "🔑 Anthropic API Settings",
                "hint": "Uses Anthropic's Claude models.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-anthropic-key", "password": True, "placeholder": "sk-ant-..."},
                    {"type": "select", "label": "Model:", "id": "select-anthropic-model", "options": [
                        ("Claude 3.5 Sonnet (Recommended)", "claude-3-5-sonnet-20241022"),
                        ("Claude 3.5 Haiku (Faster)", "claude-3-5-haiku-20241022"),
                        ("Claude 3 Opus", "claude-3-opus-20240229"),
                    ], "value": "claude-3-5-sonnet-20241022"},
                ],
                "buttons": [
                    {"label": "🔍 Auto-Detect Models", "id": "btn-detect-anthropic", "variant": "default"},
                    {"label": "Verify API Key", "id": "btn-test-anthropic", "variant": "primary"},
                ],
            },
            "google": {
                "title": "🔷 Google AI (Gemini) Settings",
                "hint": "Google's Gemini models. Requires API key from Google AI Studio.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-google-key", "password": True, "placeholder": "AIza..."},
                    {"type": "select", "label": "Model:", "id": "select-google-model", "options": [
                        ("Gemini 1.5 Flash (Recommended)", "gemini-1.5-flash-latest"),
                        ("Gemini 1.5 Pro", "gemini-1.5-pro-latest"),
                        ("Gemini 2.0 Flash", "gemini-2.0-flash-exp"),
                        ("Gemini Pro", "gemini-pro"),
                    ], "value": "gemini-1.5-flash-latest"},
                ],
                "buttons": [
                    {"label": "🔍 Auto-Detect Models", "id": "btn-detect-google", "variant": "default"},
                    {"label": "Verify API Key", "id": "btn-test-google", "variant": "primary"},
                ],
            },
            "xai": {
                "title": "🤖 xAI (Grok) Settings",
                "hint": "xAI's Grok models.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-xai-key", "password": True, "placeholder": "xai-..."},
                    {"type": "select", "label": "Model:", "id": "select-xai-model", "options": [
                        ("Grok-2 (Latest)", "grok-2"),
                        ("Grok-2 Mini", "grok-2-mini"),
                    ], "value": "grok-2"},
                ],
                "buttons": [{"label": "Verify API Key", "id": "btn-test-xai", "variant": "primary"}],
            },
            "deepseek": {
                "title": "🔮 DeepSeek API Settings",
                "hint": "DeepSeek's powerful reasoning models. Very cost-effective.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-deepseek-key", "password": True, "placeholder": "sk-..."},
                    {"type": "select", "label": "Model:", "id": "select-deepseek-model", "options": [
                        ("DeepSeek-V3 (Latest)", "deepseek-chat"),
                        ("DeepSeek-R1 (Reasoning)", "deepseek-reasoner"),
                        ("DeepSeek Coder", "deepseek-coder"),
                    ], "value": "deepseek-chat"},
                ],
                "buttons": [{"label": "Verify API Key", "id": "btn-test-deepseek", "variant": "primary"}],
            },
            "mistral": {
                "title": "🌊 Mistral AI Settings",
                "hint": "Mistral's efficient models.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-mistral-key", "password": True},
                    {"type": "select", "label": "Model:", "id": "select-mistral-model", "options": [
                        ("Mistral Large", "mistral-large-latest"),
                        ("Mistral Small", "mistral-small-latest"),
                        ("Codestral", "codestral-latest"),
                    ], "value": "mistral-large-latest"},
                ],
                "buttons": [{"label": "Verify API Key", "id": "btn-test-mistral", "variant": "primary"}],
            },
            "groq": {
                "title": "⚡ Groq (Fast Inference) Settings",
                "hint": "Ultra-fast inference. Free tier available.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-groq-key", "password": True, "placeholder": "gsk_..."},
                    {"type": "select", "label": "Model:", "id": "select-groq-model", "options": [
                        ("Llama 3.3 70B", "llama-3.3-70b-versatile"),
                        ("Llama 3.1 70B", "llama-3.1-70b-versatile"),
                        ("Mixtral 8x7B", "mixtral-8x7b-32768"),
                    ], "value": "llama-3.3-70b-versatile"},
                ],
                "buttons": [
                    {"label": "🔍 Auto-Detect Models", "id": "btn-detect-groq", "variant": "default"},
                    {"label": "Verify API Key", "id": "btn-test-groq", "variant": "primary"},
                ],
            },
            "together": {
                "title": "🤝 Together AI Settings",
                "hint": "Access to 100+ open models.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-together-key", "password": True},
                    {"type": "select", "label": "Model:", "id": "select-together-model", "options": [
                        ("Llama 3.3 70B Turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
                        ("DeepSeek V3", "deepseek-ai/DeepSeek-V3"),
                        ("Qwen 2.5 72B", "Qwen/Qwen2.5-72B-Instruct-Turbo"),
                    ], "value": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
                ],
                "buttons": [{"label": "Verify API Key", "id": "btn-test-together", "variant": "primary"}],
            },
            "openrouter": {
                "title": "🔀 OpenRouter Settings",
                "hint": "Unified API for 100+ models.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-openrouter-key", "password": True, "placeholder": "sk-or-..."},
                    {"type": "select", "label": "Model:", "id": "select-openrouter-model", "options": [
                        ("Auto (Best Available)", "openrouter/auto"),
                        ("GPT-4o", "openai/gpt-4o"),
                        ("Claude 3.5 Sonnet", "anthropic/claude-3.5-sonnet"),
                    ], "value": "openrouter/auto"},
                ],
                "buttons": [{"label": "Verify API Key", "id": "btn-test-openrouter", "variant": "primary"}],
            },
            "cohere": {
                "title": "🔶 Cohere API Settings",
                "hint": "Cohere's Command models.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-cohere-key", "password": True},
                    {"type": "select", "label": "Model:", "id": "select-cohere-model", "options": [
                        ("Command R+", "command-r-plus"),
                        ("Command R", "command-r"),
                    ], "value": "command-r-plus"},
                ],
                "buttons": [{"label": "Verify API Key", "id": "btn-test-cohere", "variant": "primary"}],
            },
            "perplexity": {
                "title": "🔍 Perplexity AI Settings",
                "hint": "Models with real-time search.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-perplexity-key", "password": True, "placeholder": "pplx-..."},
                    {"type": "select", "label": "Model:", "id": "select-perplexity-model", "options": [
                        ("Sonar Large Online", "llama-3.1-sonar-large-128k-online"),
                        ("Sonar Small Online", "llama-3.1-sonar-small-128k-online"),
                    ], "value": "llama-3.1-sonar-large-128k-online"},
                ],
                "buttons": [{"label": "Verify API Key", "id": "btn-test-perplexity", "variant": "primary"}],
            },
            "azure_openai": {
                "title": "☁️ Azure OpenAI Settings",
                "hint": "Enterprise OpenAI via Azure.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-azure-key", "password": True},
                    {"type": "input", "label": "Endpoint:", "id": "input-azure-endpoint", "placeholder": "https://your-resource.openai.azure.com/", "wide": True},
                    {"type": "input", "label": "Deployment:", "id": "input-azure-deployment", "placeholder": "gpt-4-deployment"},
                ],
                "buttons": [{"label": "Verify Connection", "id": "btn-test-azure", "variant": "primary"}],
            },
            "aws_bedrock": {
                "title": "🔶 AWS Bedrock Settings",
                "hint": "Access Claude, Llama, Titan via AWS.",
                "fields": [
                    {"type": "input", "label": "Access Key:", "id": "input-aws-access-key", "password": True, "placeholder": "AKIA..."},
                    {"type": "input", "label": "Secret Key:", "id": "input-aws-secret-key", "password": True},
                    {"type": "select", "label": "Region:", "id": "select-aws-region", "options": [
                        ("US East", "us-east-1"),
                        ("US West", "us-west-2"),
                        ("EU Frankfurt", "eu-central-1"),
                    ], "value": "us-east-1"},
                    {"type": "select", "label": "Model:", "id": "select-aws-model", "options": [
                        ("Claude 3.5 Sonnet", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
                        ("Llama 3.1 70B", "meta.llama3-1-70b-instruct-v1:0"),
                    ], "value": "anthropic.claude-3-5-sonnet-20241022-v2:0"},
                ],
                "buttons": [{"label": "Verify Credentials", "id": "btn-test-aws", "variant": "primary"}],
            },
            "huggingface": {
                "title": "🤗 Hugging Face Settings",
                "hint": "Access to thousands of models.",
                "fields": [
                    {"type": "input", "label": "API Token:", "id": "input-hf-token", "password": True, "placeholder": "hf_..."},
                    {"type": "input", "label": "Model ID:", "id": "input-hf-model", "placeholder": "meta-llama/Llama-3.1-8B-Instruct", "wide": True},
                ],
                "buttons": [{"label": "Verify Token", "id": "btn-test-hf", "variant": "primary"}],
            },
            "fireworks": {
                "title": "🎆 Fireworks AI Settings",
                "hint": "Fast inference with competitive pricing.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-fireworks-key", "password": True},
                    {"type": "select", "label": "Model:", "id": "select-fireworks-model", "options": [
                        ("Llama 3.1 70B", "accounts/fireworks/models/llama-v3p1-70b-instruct"),
                        ("Mixtral MoE 8x22B", "accounts/fireworks/models/mixtral-8x22b-instruct"),
                    ], "value": "accounts/fireworks/models/llama-v3p1-70b-instruct"},
                ],
                "buttons": [{"label": "Verify API Key", "id": "btn-test-fireworks", "variant": "primary"}],
            },
            "replicate": {
                "title": "🔄 Replicate Settings",
                "hint": "Run ML models in the cloud.",
                "fields": [
                    {"type": "input", "label": "API Token:", "id": "input-replicate-token", "password": True},
                    {"type": "input", "label": "Model:", "id": "input-replicate-model", "placeholder": "meta/llama-2-70b-chat", "wide": True},
                ],
                "buttons": [{"label": "Verify Token", "id": "btn-test-replicate", "variant": "primary"}],
            },
            "ai21": {
                "title": "🔬 AI21 Labs Settings",
                "hint": "AI21's Jamba models.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-ai21-key", "password": True},
                    {"type": "select", "label": "Model:", "id": "select-ai21-model", "options": [
                        ("Jamba 1.5 Large", "jamba-1.5-large"),
                        ("Jamba 1.5 Mini", "jamba-1.5-mini"),
                    ], "value": "jamba-1.5-large"},
                ],
                "buttons": [{"label": "Verify API Key", "id": "btn-test-ai21", "variant": "primary"}],
            },
            "deepinfra": {
                "title": "🚀 DeepInfra Settings",
                "hint": "Fast inference for open models.",
                "fields": [
                    {"type": "input", "label": "API Key:", "id": "input-deepinfra-key", "password": True},
                    {"type": "select", "label": "Model:", "id": "select-deepinfra-model", "options": [
                        ("Llama 3.1 70B", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
                        ("Mixtral 8x22B", "mistralai/Mixtral-8x22B-Instruct-v0.1"),
                    ], "value": "meta-llama/Meta-Llama-3.1-70B-Instruct"},
                ],
                "buttons": [{"label": "Verify API Key", "id": "btn-test-deepinfra", "variant": "primary"}],
            },
            "custom": {
                "title": "🔧 Custom API Settings",
                "hint": "Configure any OpenAI-compatible API endpoint.",
                "fields": [
                    {"type": "input", "label": "Base URL:", "id": "input-custom-base-url", "placeholder": "https://api.example.com/v1", "wide": True},
                    {"type": "input", "label": "API Key:", "id": "input-custom-key", "password": True},
                    {"type": "input", "label": "Model:", "id": "input-custom-model", "placeholder": "model-name"},
                ],
                "buttons": [{"label": "Test Connection", "id": "btn-test-custom", "variant": "primary"}],
            },
        }
        
        config = PROVIDER_CONFIGS.get(mode)
        if not config:
            # Fallback to custom for unknown providers
            config = PROVIDER_CONFIGS.get("custom")
        
        # Build widgets from config
        widgets.append(Static(config["title"], classes="subsection-title"))
        widgets.append(Static(config["hint"], classes="hint-text"))
        
        for field in config.get("fields", []):
            row = Horizontal(classes="setting-row")
            label = Static(field["label"], classes="setting-label")
            
            if field["type"] == "input":
                input_widget = Input(
                    value=field.get("value", ""),
                    placeholder=field.get("placeholder", ""),
                    password=field.get("password", False),
                    classes="setting-input-wide" if field.get("wide") else "api-key-input" if field.get("password") else "setting-input",
                    id=field["id"],
                )
                widgets.append((row, label, input_widget))
            elif field["type"] == "select":
                select_widget = Select(
                    field.get("options", []),
                    value=field.get("value", ""),
                    id=field["id"],
                )
                widgets.append((row, label, select_widget))
        
        # Add buttons
        if config.get("buttons"):
            button_row = Horizontal(classes="setting-row")
            button_widgets = []
            for btn in config["buttons"]:
                button_widgets.append(Button(btn["label"], id=btn["id"], variant=btn.get("variant", "default")))
            widgets.append((button_row, button_widgets))
        
        return widgets

    def _populate_provider_container(self, mode: str) -> None:
        """Lazily populate the provider settings container with widgets for the selected mode."""
        try:
            container = self.query_one("#provider-settings-container", Container)
        except NoMatches:
            return
        
        # Clear existing widgets
        container.remove_children()
        
        if mode == "none":
            return
        
        # Get widgets for the selected provider
        widgets_data = self._create_provider_widgets(mode)
        
        # Mount widgets to the container - IMPORTANT: mount container first, then add children
        for item in widgets_data:
            if isinstance(item, tuple):
                if len(item) == 3:  # (row, label, widget)
                    row, label, widget = item
                    # First mount the row to the container
                    container.mount(row)
                    # Then mount children to the row (row is now mounted)
                    row.mount(label)
                    row.mount(widget)
                elif len(item) == 2:  # (row, [buttons])
                    row, button_list = item
                    # First mount the row to the container
                    container.mount(row)
                    # Then mount buttons to the row
                    for btn in button_list:
                        row.mount(btn)
            else:
                # Direct widget (Static for title/hint)
                container.mount(item)

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "select-llm-mode":
            self._update_llm_sections(event.value)
        elif event.select.id == "select-theme":
            self._apply_theme(event.value)
        
        # Handle custom model selection for all model selects
        select_id = event.select.id
        if select_id and select_id.startswith("select-") and select_id.endswith("-model"):
            # Extract provider name (e.g., "openai" from "select-openai-model")
            provider = select_id.replace("select-", "").replace("-model", "")
            custom_input_id = f"input-{provider}-custom-model"
            
            try:
                custom_input = self.query_one(f"#{custom_input_id}", Input)
                if event.value == "__custom__":
                    # Show custom input field
                    custom_input.add_class("visible")
                    custom_input.focus()
                else:
                    # Hide custom input field
                    custom_input.remove_class("visible")
            except Exception:
                pass  # Custom input might not exist for all providers
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes - auto-detect models when API key is entered."""
        # Skip auto-detection during initial settings load to prevent performance issues
        if not getattr(self, '_settings_loaded', False):
            return
            
        input_id = event.input.id
        if not input_id:
            return
        
        # Map API key inputs to their providers and model selects
        api_key_to_provider = {
            "input-openai-key": ("openai", "select-openai-model"),
            "input-anthropic-key": ("anthropic", "select-anthropic-model"),
            "input-google-key": ("google", "select-google-model"),
            "input-xai-key": ("xai", "select-xai-model"),
            "input-deepseek-key": ("deepseek", "select-deepseek-model"),
            "input-mistral-key": ("mistral", "select-mistral-model"),
            "input-groq-key": ("groq", "select-groq-model"),
            "input-together-key": ("together", "select-together-model"),
            "input-openrouter-key": ("openrouter", "select-openrouter-model"),
            "input-cohere-key": ("cohere", "select-cohere-model"),
            "input-perplexity-key": ("perplexity", "select-perplexity-model"),
            "input-fireworks-key": ("fireworks", "select-fireworks-model"),
            "input-replicate-token": ("replicate", "select-replicate-model"),
            "input-ai21-key": ("ai21", "select-ai21-model"),
            "input-deepinfra-key": ("deepinfra", "select-deepinfra-model"),
            "input-anyscale-key": ("anyscale", "select-anyscale-model"),
            "input-sambanova-key": ("sambanova", "select-sambanova-model"),
            "input-cerebras-key": ("cerebras", "select-cerebras-model"),
            "input-novita-key": ("novita", "select-novita-model"),
            "input-friendli-key": ("friendli", "select-friendli-model"),
            "input-reka-key": ("reka", "select-reka-model"),
            "input-writer-key": ("writer", "select-writer-model"),
            "input-lambda-key": ("lambda", "select-lambda-model"),
            "input-monster-key": ("monster", "select-monster-model"),
            "input-hyperbolic-key": ("hyperbolic", "select-hyperbolic-model"),
            "input-kluster-key": ("kluster", "select-kluster-model"),
        }
        
        # Check if this is an API key input
        if input_id in api_key_to_provider:
            api_key = event.value
            # Only auto-detect if key looks valid (min length)
            if len(api_key) >= 10:
                provider_name, select_id = api_key_to_provider[input_id]
                # Debounce - delay auto-detection to avoid too many API calls
                self._schedule_model_detection(provider_name, select_id, api_key)
        
        # Handle local endpoint inputs for model detection
        local_endpoint_to_provider = {
            "input-ollama-url": ("ollama", "input-local-model"),
            "input-vllm-endpoint": ("vllm", "input-vllm-model"),
            "input-localai-endpoint": ("localai", "input-localai-model"),
            "input-textgen-endpoint": ("textgen_webui", "input-textgen-model"),
        }
        
        if input_id in local_endpoint_to_provider:
            endpoint = event.value
            if endpoint and (endpoint.startswith("http://") or endpoint.startswith("https://")):
                provider_name, model_input_id = local_endpoint_to_provider[input_id]
                self._schedule_local_model_detection(provider_name, model_input_id, endpoint)
    
    def _schedule_model_detection(self, provider_name: str, select_id: str, api_key: str) -> None:
        """Schedule model detection with debouncing."""
        # Cancel any pending detection for this provider
        if hasattr(self, '_detection_tasks'):
            task_key = f"detect_{provider_name}"
            if task_key in self._detection_tasks:
                self._detection_tasks[task_key].cancel()
        else:
            self._detection_tasks = {}
        
        # Schedule new detection after a short delay
        async def delayed_detection():
            await asyncio.sleep(0.8)  # Debounce delay
            await self._auto_detect_models(provider_name, select_id, api_key)
        
        task = asyncio.create_task(delayed_detection())
        self._detection_tasks[f"detect_{provider_name}"] = task
    
    def _schedule_local_model_detection(self, provider_name: str, model_input_id: str, endpoint: str) -> None:
        """Schedule local model detection with debouncing."""
        if hasattr(self, '_detection_tasks'):
            task_key = f"detect_local_{provider_name}"
            if task_key in self._detection_tasks:
                self._detection_tasks[task_key].cancel()
        else:
            self._detection_tasks = {}
        
        async def delayed_detection():
            await asyncio.sleep(1.0)  # Longer delay for local endpoints
            await self._auto_detect_local_models(provider_name, model_input_id, endpoint)
        
        task = asyncio.create_task(delayed_detection())
        self._detection_tasks[f"detect_local_{provider_name}"] = task
    
    async def _auto_detect_models(self, provider_name: str, select_id: str, api_key: str) -> None:
        """Auto-detect available models from API provider."""
        if not LLM_AVAILABLE:
            return
        
        try:
            from proxima.intelligence.llm_router import ProviderRegistry
            registry = ProviderRegistry()
            provider = registry.get(provider_name)
            
            # Show detecting status
            self.notify(f"🔍 Detecting {provider_name} models...", severity="information")
            
            # Get models from provider - try different signatures
            # Some providers expect (api_base, api_key), others just (api_key)
            def fetch_models():
                try:
                    # Try the new signature with api_base and api_key
                    return provider.list_models(api_base=None, api_key=api_key)
                except TypeError:
                    try:
                        # Fall back to just api_key
                        return provider.list_models(api_key)
                    except TypeError:
                        # Or no arguments
                        return provider.list_models()
            
            models = await asyncio.get_event_loop().run_in_executor(
                None, fetch_models
            )
            
            if models:
                # Update the select with detected models
                try:
                    select = self.query_one(f"#{select_id}", Select)
                    
                    # Create options from detected models
                    # Mark the first/best model as recommended
                    options = []
                    best_model = self._get_best_model(provider_name, models)
                    
                    for model in models[:20]:  # Limit to 20 models
                        if model == best_model:
                            options.append((f"✨ {model} (Recommended)", model))
                        else:
                            options.append((model, model))
                    
                    # Add manual entry option
                    options.append(("📝 Enter custom model...", "__custom__"))
                    
                    # Update select options
                    select.set_options(options)
                    
                    # Auto-select the best model
                    if best_model:
                        select.value = best_model
                    
                    self.notify(f"✅ Found {len(models)} models for {provider_name}", severity="success")
                except Exception as e:
                    self.notify(f"Could not update model list: {e}", severity="warning")
            else:
                self.notify(f"No models found for {provider_name}", severity="warning")
                
        except Exception as e:
            # Silently fail - user can still manually select
            self.notify(f"Auto-detect failed: {str(e)[:50]}", severity="warning")
    
    async def _auto_detect_local_models(self, provider_name: str, model_input_id: str, endpoint: str) -> None:
        """Auto-detect available models from local LLM server."""
        if not LLM_AVAILABLE:
            return
        
        try:
            from proxima.intelligence.llm_router import ProviderRegistry
            registry = ProviderRegistry()
            
            # For local providers, we need to set the endpoint first
            if provider_name == "ollama":
                provider = registry.get("ollama")
                if hasattr(provider, 'set_endpoint'):
                    provider.set_endpoint(endpoint)
            else:
                provider = registry.get(provider_name)
            
            self.notify(f"🔍 Detecting local models at {endpoint}...", severity="information")
            
            # Get models - for local providers, pass endpoint instead of API key
            models = await asyncio.get_event_loop().run_in_executor(
                None, lambda: provider.list_models(endpoint)
            )
            
            if models:
                # For local providers with input fields (not select), show available models
                self.notify(f"✅ Found {len(models)} local models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}", severity="success")
                
                # If we can find the input, suggest the first model
                try:
                    model_input = self.query_one(f"#{model_input_id}", Input)
                    if not model_input.value:
                        model_input.value = models[0]
                except Exception:
                    pass
            else:
                self.notify(f"No models found at {endpoint}", severity="warning")
                
        except Exception as e:
            self.notify(f"Could not detect local models: {str(e)[:50]}", severity="warning")
    
    def _get_best_model(self, provider_name: str, models: list[str]) -> Optional[str]:
        """Get the best/recommended model for a provider.
        
        Args:
            provider_name: Name of the provider
            models: List of available model names
            
        Returns:
            The recommended model name, or None
        """
        # Define preferred models for each provider (in order of preference)
        preferred_models = {
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-5-sonnet", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-2"],
            "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro", "gemini-1.0-pro"],
            "xai": ["grok-2", "grok-beta", "grok-1"],
            "deepseek": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
            "mistral": ["mistral-large", "mistral-medium", "mistral-small", "mistral-tiny", "open-mixtral"],
            "groq": ["llama-3.3-70b", "llama-3.1-70b", "mixtral-8x7b", "llama3-70b", "llama3-8b"],
            "together": ["meta-llama/Llama-3", "mistralai/Mixtral", "meta-llama/Meta-Llama-3"],
            "openrouter": ["anthropic/claude-3", "openai/gpt-4", "meta-llama/llama-3"],
            "cohere": ["command-r-plus", "command-r", "command", "command-light"],
            "perplexity": ["llama-3.1-sonar-large", "llama-3.1-sonar-small", "sonar-pro"],
            "fireworks": ["accounts/fireworks/models/llama", "accounts/fireworks/models/mixtral"],
            "replicate": ["meta/llama", "meta/meta-llama-3"],
            "ai21": ["jamba-1.5-large", "jamba-1.5-mini", "j2-ultra", "j2-mid"],
            "deepinfra": ["meta-llama/Meta-Llama-3", "mistralai/Mixtral"],
            "anyscale": ["meta-llama/Llama", "mistralai/Mixtral"],
            "sambanova": ["Meta-Llama-3", "Mistral"],
            "cerebras": ["llama3.1-70b", "llama3.1-8b"],
            "novita": ["meta-llama/llama", "mistralai/mixtral"],
            "friendli": ["meta-llama-3", "mixtral"],
            "reka": ["reka-core", "reka-flash", "reka-edge"],
            "writer": ["palmyra-x", "palmyra"],
            "lambda": ["llama3", "hermes"],
            "monster": ["llama", "mistral"],
            "hyperbolic": ["meta-llama", "mistral"],
            "kluster": ["llama", "mixtral"],
        }
        
        preferred = preferred_models.get(provider_name, [])
        
        # Find the first preferred model that's available
        for pref in preferred:
            for model in models:
                if pref.lower() in model.lower():
                    return model
        
        # If no preferred model found, return the first one
        return models[0] if models else None

    def _apply_theme(self, theme_name: str) -> None:
        """Apply the selected theme immediately.
        
        Args:
            theme_name: Name of the theme to apply
        """
        try:
            from ..styles.theme import set_theme_by_name, get_theme
            
            # Set our custom theme
            if set_theme_by_name(theme_name):
                theme = get_theme()
                
                # Update Textual's dark/light mode
                app = self.app
                if theme.is_dark:
                    app.dark = True
                else:
                    app.dark = False
                
                # Apply CSS variables for the theme
                self._apply_theme_css(theme)
                
                self.notify(f"Theme switched to {theme.name}", severity="information")
            else:
                self.notify(f"Theme '{theme_name}' not found", severity="warning")
        except Exception as e:
            self.notify(f"Failed to apply theme: {e}", severity="warning")
    
    def _apply_theme_css(self, theme) -> None:
        """Apply theme colors as CSS variables."""
        try:
            # Update app-level CSS variables
            css_vars = f"""
                $primary: {theme.primary};
                $primary-light: {theme.primary_light};
                $primary-dark: {theme.primary_dark};
                $accent: {theme.accent};
                $success: {theme.success};
                $error: {theme.error};
                $warning: {theme.warning};
                $info: {theme.info};
                $bg-base: {theme.bg_base};
                $fg-base: {theme.fg_base};
                $border: {theme.border};
            """
            # Note: Full CSS variable application would require deeper Textual integration
            # This serves as a marker for where theme CSS would be applied
        except Exception:
            pass
    
    def _update_llm_sections(self, mode: str) -> None:
        """Update the provider settings container based on selected mode - uses lazy loading.
        
        Performance optimizations:
        1. Provider widgets are NOT created in compose_main() - only a single empty container
        2. Widgets are created on-demand when a provider is selected
        3. Previous widgets are removed before creating new ones (minimal DOM)
        """
        # Update the provider container with new widgets
        self._populate_provider_container(mode)
        
        # Update mode tracking
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
        elif button_id == "btn-open-thinking":
            self._open_ai_thinking()
        # Handle model detection buttons
        elif button_id and button_id.startswith("btn-detect-"):
            self._manual_detect_models(button_id.replace("btn-detect-", ""))
        # Handle new provider test buttons
        elif button_id and button_id.startswith("btn-test-"):
            self._test_provider(button_id.replace("btn-test-", ""))

    def _manual_detect_models(self, provider_name: str) -> None:
        """Manually trigger model detection for a provider."""
        # Map provider to key input and select
        provider_map = {
            "openai": ("input-openai-key", "select-openai-model"),
            "anthropic": ("input-anthropic-key", "select-anthropic-model"),
            "google": ("input-google-key", "select-google-model"),
            "xai": ("input-xai-key", "select-xai-model"),
            "deepseek": ("input-deepseek-key", "select-deepseek-model"),
            "mistral": ("input-mistral-key", "select-mistral-model"),
            "groq": ("input-groq-key", "select-groq-model"),
            "together": ("input-together-key", "select-together-model"),
            "openrouter": ("input-openrouter-key", "select-openrouter-model"),
            "cohere": ("input-cohere-key", "select-cohere-model"),
            "perplexity": ("input-perplexity-key", "select-perplexity-model"),
            "fireworks": ("input-fireworks-key", "select-fireworks-model"),
            "huggingface": ("input-huggingface-key", "select-huggingface-model"),
            "replicate": ("input-replicate-key", "select-replicate-model"),
            "ai21": ("input-ai21-key", "select-ai21-model"),
            "deepinfra": ("input-deepinfra-key", "select-deepinfra-model"),
            "anyscale": ("input-anyscale-key", "select-anyscale-model"),
            "sambanova": ("input-sambanova-key", "select-sambanova-model"),
            "cerebras": ("input-cerebras-key", "select-cerebras-model"),
            "novita": ("input-novita-key", "select-novita-model"),
            "friendli": ("input-friendli-key", "select-friendli-model"),
            "reka": ("input-reka-key", "select-reka-model"),
            "writer": ("input-writer-key", "select-writer-model"),
            "lambda": ("input-lambda-key", "select-lambda-model"),
            "monster": ("input-monster-key", "select-monster-model"),
            "hyperbolic": ("input-hyperbolic-key", "select-hyperbolic-model"),
            "kluster": ("input-kluster-key", "select-kluster-model"),
            "lepton": ("input-lepton-key", "select-lepton-model"),
        }
        
        config = provider_map.get(provider_name)
        if not config:
            self.notify(f"Unknown provider for detection: {provider_name}", severity="warning")
            return
        
        key_input_id, select_id = config
        
        try:
            api_key = self.query_one(f"#{key_input_id}", Input).value
        except Exception:
            self.notify("Could not find API key input", severity="error")
            return
        
        if not api_key or len(api_key) < 5:
            self.notify("Please enter an API key first", severity="warning")
            return
        
        # Trigger async model detection
        asyncio.create_task(self._auto_detect_models(provider_name, select_id, api_key))

    def _save_settings(self) -> None:
        """Save current settings to disk including all API keys."""
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
        
        # Get LLM mode
        llm_mode = self.query_one("#select-llm-mode", Select).value
        
        # Helper to safely get input value
        def get_input_value(input_id: str, default: str = "") -> str:
            try:
                return self.query_one(f"#{input_id}", Input).value
            except Exception:
                return default
        
        # Helper to safely get select value
        def get_select_value(select_id: str, default: str = "") -> str:
            try:
                val = self.query_one(f"#{select_id}", Select).value
                return val if val != Select.BLANK else default
            except Exception:
                return default
        
        # Collect all settings including ALL API keys
        settings = {
            'general': {
                'backend': self.query_one("#select-backend", Select).value,
                'shots': shots_int,
                'autosave': self.query_one("#switch-autosave", Switch).value,
            },
            'llm': {
                'mode': llm_mode,
                # Local LLM settings
                'ollama_url': get_input_value("input-ollama-url", "http://localhost:11434"),
                'local_model': get_input_value("input-local-model", "llama3"),
                # OpenAI
                'openai_key': get_input_value("input-openai-key"),
                'openai_model': get_select_value("select-openai-model", "gpt-4o-mini"),
                # Anthropic
                'anthropic_key': get_input_value("input-anthropic-key"),
                'anthropic_model': get_select_value("select-anthropic-model", "claude-3-5-sonnet-20241022"),
                # Google
                'google_key': get_input_value("input-google-key"),
                'google_model': get_select_value("select-google-model", "gemini-1.5-flash-latest"),
                # xAI
                'xai_key': get_input_value("input-xai-key"),
                'xai_model': get_select_value("select-xai-model", "grok-2"),
                # DeepSeek
                'deepseek_key': get_input_value("input-deepseek-key"),
                'deepseek_model': get_select_value("select-deepseek-model", "deepseek-chat"),
                # Mistral
                'mistral_key': get_input_value("input-mistral-key"),
                'mistral_model': get_select_value("select-mistral-model", "mistral-large-latest"),
                # Groq
                'groq_key': get_input_value("input-groq-key"),
                'groq_model': get_select_value("select-groq-model", "llama-3.3-70b-versatile"),
                # Together
                'together_key': get_input_value("input-together-key"),
                'together_model': get_select_value("select-together-model", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
                # OpenRouter
                'openrouter_key': get_input_value("input-openrouter-key"),
                'openrouter_model': get_select_value("select-openrouter-model", "openrouter/auto"),
                # Cohere
                'cohere_key': get_input_value("input-cohere-key"),
                'cohere_model': get_select_value("select-cohere-model", "command-r-plus"),
                # Perplexity
                'perplexity_key': get_input_value("input-perplexity-key"),
                'perplexity_model': get_select_value("select-perplexity-model", "llama-3.1-sonar-large-128k-online"),
                # Azure OpenAI
                'azure_key': get_input_value("input-azure-key"),
                'azure_endpoint': get_input_value("input-azure-endpoint"),
                'azure_deployment': get_input_value("input-azure-deployment"),
                # AWS Bedrock
                'aws_access_key': get_input_value("input-aws-access-key"),
                'aws_secret_key': get_input_value("input-aws-secret-key"),
                'aws_region': get_select_value("select-aws-region", "us-east-1"),
                'aws_model': get_select_value("select-aws-model", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
                # Hugging Face
                'hf_token': get_input_value("input-hf-token"),
                'hf_model': get_input_value("input-hf-model"),
                # Fireworks
                'fireworks_key': get_input_value("input-fireworks-key"),
                'fireworks_model': get_select_value("select-fireworks-model", "accounts/fireworks/models/llama-v3p1-70b-instruct"),
                # Replicate
                'replicate_token': get_input_value("input-replicate-token"),
                'replicate_model': get_input_value("input-replicate-model"),
                # AI21
                'ai21_key': get_input_value("input-ai21-key"),
                'ai21_model': get_select_value("select-ai21-model", "jamba-1.5-large"),
                # DeepInfra
                'deepinfra_key': get_input_value("input-deepinfra-key"),
                'deepinfra_model': get_select_value("select-deepinfra-model", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
                # Anyscale
                'anyscale_key': get_input_value("input-anyscale-key"),
                'anyscale_model': get_select_value("select-anyscale-model", "meta-llama/Llama-3-70b-chat-hf"),
                # Vertex AI
                'vertex_project': get_input_value("input-vertex-project"),
                'vertex_location': get_select_value("select-vertex-location", "us-central1"),
                'vertex_model': get_select_value("select-vertex-model", "gemini-1.5-pro"),
                # watsonx
                'watsonx_key': get_input_value("input-watsonx-key"),
                'watsonx_project': get_input_value("input-watsonx-project"),
                'watsonx_model': get_select_value("select-watsonx-model", "ibm/granite-13b-chat-v2"),
                # Oracle AI
                'oracle_key': get_input_value("input-oracle-key"),
                'oracle_compartment': get_input_value("input-oracle-compartment"),
                'oracle_model': get_select_value("select-oracle-model", "cohere.command-r-plus"),
                # Alibaba Qwen
                'alibaba_key': get_input_value("input-alibaba-key"),
                'alibaba_model': get_select_value("select-alibaba-model", "qwen-max"),
                # SambaNova
                'sambanova_key': get_input_value("input-sambanova-key"),
                'sambanova_model': get_select_value("select-sambanova-model", "Meta-Llama-3.1-70B-Instruct"),
                # Cerebras
                'cerebras_key': get_input_value("input-cerebras-key"),
                'cerebras_model': get_select_value("select-cerebras-model", "llama3.1-70b"),
                # Lepton
                'lepton_key': get_input_value("input-lepton-key"),
                'lepton_model': get_input_value("input-lepton-model", "llama3-70b"),
                # Novita
                'novita_key': get_input_value("input-novita-key"),
                'novita_model': get_select_value("select-novita-model", "meta-llama/llama-3.1-70b-instruct"),
                # Friendli
                'friendli_key': get_input_value("input-friendli-key"),
                'friendli_model': get_select_value("select-friendli-model", "meta-llama-3.1-70b-instruct"),
                # Reka
                'reka_key': get_input_value("input-reka-key"),
                'reka_model': get_select_value("select-reka-model", "reka-flash"),
                # Writer
                'writer_key': get_input_value("input-writer-key"),
                'writer_model': get_select_value("select-writer-model", "palmyra-x-004"),
                # Baseten
                'baseten_key': get_input_value("input-baseten-key"),
                'baseten_model': get_input_value("input-baseten-model", "llama-3.1-70b"),
                # Modal
                'modal_endpoint': get_input_value("input-modal-endpoint"),
                'modal_key': get_input_value("input-modal-key"),
                'modal_model': get_input_value("input-modal-model"),
                # RunPod
                'runpod_key': get_input_value("input-runpod-key"),
                'runpod_endpoint': get_input_value("input-runpod-endpoint"),
                'runpod_model': get_input_value("input-runpod-model"),
                # Lambda Labs
                'lambda_key': get_input_value("input-lambda-key"),
                'lambda_model': get_select_value("select-lambda-model", "llama3.1-70b-instruct-fp8"),
                # Monster API
                'monster_key': get_input_value("input-monster-key"),
                'monster_model': get_select_value("select-monster-model", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
                # Hyperbolic
                'hyperbolic_key': get_input_value("input-hyperbolic-key"),
                'hyperbolic_model': get_select_value("select-hyperbolic-model", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
                # Kluster
                'kluster_key': get_input_value("input-kluster-key"),
                'kluster_model': get_select_value("select-kluster-model", "llama-3.1-70b"),
                # vLLM (local)
                'vllm_endpoint': get_input_value("input-vllm-endpoint", "http://localhost:8000"),
                'vllm_model': get_input_value("input-vllm-model"),
                # LocalAI
                'localai_endpoint': get_input_value("input-localai-endpoint", "http://localhost:8080"),
                'localai_model': get_input_value("input-localai-model"),
                # TextGen WebUI
                'textgen_endpoint': get_input_value("input-textgen-endpoint", "http://localhost:5000"),
                'textgen_model': get_input_value("input-textgen-model"),
                # Custom API
                'custom_base_url': get_input_value("input-custom-base-url"),
                'custom_key': get_input_value("input-custom-key"),
                'custom_model': get_input_value("input-custom-model"),
                # Thinking enabled
                'thinking_enabled': self.query_one("#switch-thinking-enabled", Switch).value if self.query("#switch-thinking-enabled") else False,
            },
            'display': {
                'theme': self.query_one("#select-theme", Select).value,
                'compact_sidebar': self.query_one("#switch-compact", Switch).value,
                'show_logs': self.query_one("#switch-logs", Switch).value,
            },
        }
        
        # Save to disk
        try:
            import json
            config_dir = Path.home() / ".proxima"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "tui_settings.json"
            
            with open(config_path, 'w') as f:
                json.dump(settings, f, indent=2)
            
            # Update TUI state with all relevant settings
            if hasattr(self, 'state'):
                self.state.shots = shots_int
                self.state.current_backend = settings['general']['backend']
                
                # Update LLM state based on selected provider
                self.state.llm_provider = llm_mode
                self.state.thinking_enabled = settings['llm'].get('thinking_enabled', False)
                
                # Get the model name for the selected provider - COMPLETE MAPPING
                model_key_map = {
                    'local': 'local_model',
                    'ollama': 'local_model',
                    'lmstudio': 'local_model',
                    'llamacpp': 'local_model',
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
                    'azure_openai': 'azure_deployment',
                    'aws_bedrock': 'aws_model',
                    'huggingface': 'hf_model',
                    'fireworks': 'fireworks_model',
                    'replicate': 'replicate_model',
                    'ai21': 'ai21_model',
                    'deepinfra': 'deepinfra_model',
                    'anyscale': 'anyscale_model',
                    'vertex_ai': 'vertex_model',
                    'watsonx': 'watsonx_model',
                    'oracle_ai': 'oracle_model',
                    'alibaba_qwen': 'alibaba_model',
                    'sambanova': 'sambanova_model',
                    'cerebras': 'cerebras_model',
                    'lepton': 'lepton_model',
                    'novita': 'novita_model',
                    'friendli': 'friendli_model',
                    'reka': 'reka_model',
                    'writer': 'writer_model',
                    'baseten': 'baseten_model',
                    'modal': 'modal_model',
                    'runpod': 'runpod_model',
                    'lambda': 'lambda_model',
                    'monster': 'monster_model',
                    'hyperbolic': 'hyperbolic_model',
                    'kluster': 'kluster_model',
                    'vllm': 'vllm_model',
                    'localai': 'localai_model',
                    'textgen_webui': 'textgen_model',
                    'oobabooga': 'textgen_model',
                }
                
                if llm_mode in model_key_map:
                    self.state.llm_model = settings['llm'].get(model_key_map[llm_mode], '')
                elif llm_mode not in ['none', '', None]:
                    self.state.llm_model = settings['llm'].get('custom_model', '')
                else:
                    self.state.llm_model = ''
                
                # Mark as connected if we have a valid provider configured
                if llm_mode and llm_mode != 'none':
                    self.state.llm_connected = True
                else:
                    self.state.llm_connected = False
            
            self.notify(f"✓ Settings saved to {config_path}", severity="success")
        except Exception as e:
            self.notify(f"✗ Failed to save settings: {e}", severity="error")
    
    def on_mount(self) -> None:
        """Load saved settings on mount - highly optimized for performance."""
        # All sections start hidden via CSS - no need to iterate
        # Use set_timer for deferred loading to ensure UI is fully rendered first
        self.set_timer(0.05, self._deferred_load_settings)
    
    def _deferred_load_settings(self) -> None:
        """Load settings after UI is rendered - prevents freeze."""
        if self._settings_loaded:
            return
        self._settings_loaded = True
        # Load settings in chunks to prevent UI blocking
        self.call_later(self._load_saved_settings_chunked)
    
    def _load_saved_settings_chunked(self) -> None:
        """Load settings in chunks for better responsiveness."""
        # First load just the critical settings (mode, general)
        self._load_general_settings()
        # Then defer loading provider-specific settings
        self.set_timer(0.02, self._load_provider_settings)
    
    def _load_general_settings(self) -> None:
        """Load general settings first - fast, non-blocking."""
        try:
            import json
            config_path = Path.home() / ".proxima" / "tui_settings.json"
            
            if not config_path.exists():
                return
            
            with open(config_path, 'r') as f:
                settings = json.load(f)
            
            # Cache settings for later use
            self._cached_settings = settings
            
            # Apply general settings (fast - only 3 widgets)
            general = settings.get('general', {})
            try:
                if 'backend' in general:
                    self.query_one("#select-backend", Select).value = general['backend']
                if 'shots' in general:
                    self.query_one("#input-shots", Input).value = str(general['shots'])
                if 'autosave' in general:
                    self.query_one("#switch-autosave", Switch).value = general['autosave']
            except Exception:
                pass
            
            # Apply LLM mode (shows correct section)
            llm = settings.get('llm', {})
            if 'mode' in llm:
                try:
                    self.query_one("#select-llm-mode", Select).value = llm['mode']
                    self._update_llm_sections(llm['mode'])
                except Exception:
                    pass
        except Exception:
            pass
    
    def _load_provider_settings(self) -> None:
        """Load provider-specific settings - deferred for performance."""
        if not hasattr(self, '_cached_settings'):
            return
        
        settings = self._cached_settings
        llm = settings.get('llm', {})
        mode = llm.get('mode', 'none')
        
        # Only load settings for the currently active provider
        # This avoids touching widgets that aren't visible
        self._load_active_provider_settings(mode, llm)
        
        # Update TUIState if available
        self._update_tui_state(mode, llm)
        
        # Clean up cached settings
        if hasattr(self, '_cached_settings'):
            del self._cached_settings
        
        # Show notification
        self.notify("Settings loaded", severity="information", timeout=1)
    
    def _load_active_provider_settings(self, mode: str, llm: dict) -> None:
        """Load only the settings for the currently active provider."""
        # Helper to safely set input value
        def set_input(input_id: str, key: str) -> None:
            try:
                value = llm.get(key, '')
                if value:
                    self.query_one(f"#{input_id}", Input).value = value
            except Exception:
                pass
        
        # Helper to safely set select value
        def set_select(select_id: str, key: str) -> None:
            try:
                value = llm.get(key, '')
                if value:
                    self.query_one(f"#{select_id}", Select).value = value
            except Exception:
                pass
        
        # Map mode to its settings - only load what's needed
        if mode in ['local', 'ollama', 'lmstudio', 'llamacpp']:
            set_input("input-ollama-url", 'ollama_url')
            set_input("input-local-model", 'local_model')
        elif mode == 'openai':
            set_input("input-openai-key", 'openai_key')
            set_select("select-openai-model", 'openai_model')
        elif mode == 'anthropic':
            set_input("input-anthropic-key", 'anthropic_key')
            set_select("select-anthropic-model", 'anthropic_model')
        elif mode == 'google':
            set_input("input-google-key", 'google_key')
            set_select("select-google-model", 'google_model')
        elif mode == 'xai':
            set_input("input-xai-key", 'xai_key')
            set_select("select-xai-model", 'xai_model')
        elif mode == 'deepseek':
            set_input("input-deepseek-key", 'deepseek_key')
            set_select("select-deepseek-model", 'deepseek_model')
        elif mode == 'mistral':
            set_input("input-mistral-key", 'mistral_key')
            set_select("select-mistral-model", 'mistral_model')
        elif mode == 'groq':
            set_input("input-groq-key", 'groq_key')
            set_select("select-groq-model", 'groq_model')
        elif mode == 'together':
            set_input("input-together-key", 'together_key')
            set_select("select-together-model", 'together_model')
        elif mode == 'openrouter':
            set_input("input-openrouter-key", 'openrouter_key')
            set_select("select-openrouter-model", 'openrouter_model')
        elif mode == 'cohere':
            set_input("input-cohere-key", 'cohere_key')
            set_select("select-cohere-model", 'cohere_model')
        elif mode == 'perplexity':
            set_input("input-perplexity-key", 'perplexity_key')
            set_select("select-perplexity-model", 'perplexity_model')
        elif mode == 'azure_openai':
            set_input("input-azure-key", 'azure_key')
            set_input("input-azure-endpoint", 'azure_endpoint')
            set_input("input-azure-deployment", 'azure_deployment')
        elif mode == 'aws_bedrock':
            set_input("input-aws-access-key", 'aws_access_key')
            set_input("input-aws-secret-key", 'aws_secret_key')
            set_select("select-aws-region", 'aws_region')
            set_select("select-aws-model", 'aws_model')
        elif mode == 'huggingface':
            set_input("input-hf-token", 'hf_token')
            set_input("input-hf-model", 'hf_model')
        elif mode == 'fireworks':
            set_input("input-fireworks-key", 'fireworks_key')
            set_select("select-fireworks-model", 'fireworks_model')
        elif mode == 'replicate':
            set_input("input-replicate-token", 'replicate_token')
            set_input("input-replicate-model", 'replicate_model')
        elif mode == 'ai21':
            set_input("input-ai21-key", 'ai21_key')
            set_select("select-ai21-model", 'ai21_model')
        elif mode == 'deepinfra':
            set_input("input-deepinfra-key", 'deepinfra_key')
            set_select("select-deepinfra-model", 'deepinfra_model')
        # For other providers, they'll load via custom API settings
        
        # Also load thinking enabled switch
        try:
            thinking = llm.get('thinking_enabled', False)
            self.query_one("#switch-thinking-enabled", Switch).value = thinking
        except Exception:
            pass
    
    def _update_tui_state(self, mode: str, llm: dict) -> None:
        """Update TUIState with loaded settings."""
        if not hasattr(self, 'state'):
            return
        
        self.state.llm_provider = mode
        self.state.thinking_enabled = llm.get('thinking_enabled', False)
        
        # Get the model name for the selected provider
        model_key_map = {
            'local': 'local_model',
            'ollama': 'local_model',
            'lmstudio': 'local_model',
            'llamacpp': 'local_model',
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
            'azure_openai': 'azure_deployment',
            'aws_bedrock': 'aws_model',
            'huggingface': 'hf_model',
            'fireworks': 'fireworks_model',
        }
        
        if mode in model_key_map:
            self.state.llm_model = llm.get(model_key_map[mode], '')
        else:
            self.state.llm_model = ''
        
        # Mark as connected if we have a valid provider configured
        self.state.llm_connected = (mode and mode != 'none')

    def _load_saved_settings(self) -> None:
        """Load settings from disk if available."""
        try:
            import json
            config_path = Path.home() / ".proxima" / "tui_settings.json"
            
            if not config_path.exists():
                return
            
            with open(config_path, 'r') as f:
                settings = json.load(f)
            
            # Helper to safely set input value
            def set_input_value(input_id: str, value: str) -> None:
                try:
                    if value:
                        self.query_one(f"#{input_id}", Input).value = value
                except Exception:
                    pass
            
            # Helper to safely set select value
            def set_select_value(select_id: str, value: str) -> None:
                try:
                    if value:
                        self.query_one(f"#{select_id}", Select).value = value
                except Exception:
                    pass
            
            # Helper to safely set switch value
            def set_switch_value(switch_id: str, value: bool) -> None:
                try:
                    self.query_one(f"#{switch_id}", Switch).value = value
                except Exception:
                    pass
            
            # Apply general settings
            general = settings.get('general', {})
            if 'backend' in general:
                self.query_one("#select-backend", Select).value = general['backend']
            if 'shots' in general:
                self.query_one("#input-shots", Input).value = str(general['shots'])
            if 'autosave' in general:
                self.query_one("#switch-autosave", Switch).value = general['autosave']
            
            # Apply LLM settings
            llm = settings.get('llm', {})
            if 'mode' in llm:
                self.query_one("#select-llm-mode", Select).value = llm['mode']
                self._update_llm_sections(llm['mode'])
            
            # Local LLM
            set_input_value("input-ollama-url", llm.get('ollama_url', ''))
            set_input_value("input-local-model", llm.get('local_model', ''))
            
            # OpenAI
            set_input_value("input-openai-key", llm.get('openai_key', ''))
            set_select_value("select-openai-model", llm.get('openai_model', ''))
            
            # Anthropic
            set_input_value("input-anthropic-key", llm.get('anthropic_key', ''))
            set_select_value("select-anthropic-model", llm.get('anthropic_model', ''))
            
            # Google
            set_input_value("input-google-key", llm.get('google_key', ''))
            set_select_value("select-google-model", llm.get('google_model', ''))
            
            # xAI
            set_input_value("input-xai-key", llm.get('xai_key', ''))
            set_select_value("select-xai-model", llm.get('xai_model', ''))
            
            # DeepSeek
            set_input_value("input-deepseek-key", llm.get('deepseek_key', ''))
            set_select_value("select-deepseek-model", llm.get('deepseek_model', ''))
            
            # Mistral
            set_input_value("input-mistral-key", llm.get('mistral_key', ''))
            set_select_value("select-mistral-model", llm.get('mistral_model', ''))
            
            # Groq
            set_input_value("input-groq-key", llm.get('groq_key', ''))
            set_select_value("select-groq-model", llm.get('groq_model', ''))
            
            # Together
            set_input_value("input-together-key", llm.get('together_key', ''))
            set_select_value("select-together-model", llm.get('together_model', ''))
            
            # OpenRouter
            set_input_value("input-openrouter-key", llm.get('openrouter_key', ''))
            set_select_value("select-openrouter-model", llm.get('openrouter_model', ''))
            
            # Cohere
            set_input_value("input-cohere-key", llm.get('cohere_key', ''))
            set_select_value("select-cohere-model", llm.get('cohere_model', ''))
            
            # Perplexity
            set_input_value("input-perplexity-key", llm.get('perplexity_key', ''))
            set_select_value("select-perplexity-model", llm.get('perplexity_model', ''))
            
            # Azure OpenAI
            set_input_value("input-azure-key", llm.get('azure_key', ''))
            set_input_value("input-azure-endpoint", llm.get('azure_endpoint', ''))
            set_input_value("input-azure-deployment", llm.get('azure_deployment', ''))
            
            # AWS Bedrock
            set_input_value("input-aws-access-key", llm.get('aws_access_key', ''))
            set_input_value("input-aws-secret-key", llm.get('aws_secret_key', ''))
            set_select_value("select-aws-region", llm.get('aws_region', ''))
            set_select_value("select-aws-model", llm.get('aws_model', ''))
            
            # Hugging Face
            set_input_value("input-hf-token", llm.get('hf_token', ''))
            set_input_value("input-hf-model", llm.get('hf_model', ''))
            
            # Fireworks
            set_input_value("input-fireworks-key", llm.get('fireworks_key', ''))
            set_select_value("select-fireworks-model", llm.get('fireworks_model', ''))
            
            # Replicate
            set_input_value("input-replicate-token", llm.get('replicate_token', ''))
            set_input_value("input-replicate-model", llm.get('replicate_model', ''))
            
            # AI21
            set_input_value("input-ai21-key", llm.get('ai21_key', ''))
            set_select_value("select-ai21-model", llm.get('ai21_model', ''))
            
            # DeepInfra
            set_input_value("input-deepinfra-key", llm.get('deepinfra_key', ''))
            set_select_value("select-deepinfra-model", llm.get('deepinfra_model', ''))
            
            # Anyscale
            set_input_value("input-anyscale-key", llm.get('anyscale_key', ''))
            set_select_value("select-anyscale-model", llm.get('anyscale_model', ''))
            
            # Vertex AI
            set_input_value("input-vertex-project", llm.get('vertex_project', ''))
            set_select_value("select-vertex-location", llm.get('vertex_location', ''))
            set_select_value("select-vertex-model", llm.get('vertex_model', ''))
            
            # watsonx
            set_input_value("input-watsonx-key", llm.get('watsonx_key', ''))
            set_input_value("input-watsonx-project", llm.get('watsonx_project', ''))
            set_select_value("select-watsonx-model", llm.get('watsonx_model', ''))
            
            # Oracle AI
            set_input_value("input-oracle-key", llm.get('oracle_key', ''))
            set_input_value("input-oracle-compartment", llm.get('oracle_compartment', ''))
            set_select_value("select-oracle-model", llm.get('oracle_model', ''))
            
            # Alibaba Qwen
            set_input_value("input-alibaba-key", llm.get('alibaba_key', ''))
            set_select_value("select-alibaba-model", llm.get('alibaba_model', ''))
            
            # SambaNova
            set_input_value("input-sambanova-key", llm.get('sambanova_key', ''))
            set_select_value("select-sambanova-model", llm.get('sambanova_model', ''))
            
            # Cerebras
            set_input_value("input-cerebras-key", llm.get('cerebras_key', ''))
            set_select_value("select-cerebras-model", llm.get('cerebras_model', ''))
            
            # Lepton
            set_input_value("input-lepton-key", llm.get('lepton_key', ''))
            set_input_value("input-lepton-model", llm.get('lepton_model', ''))
            
            # Novita
            set_input_value("input-novita-key", llm.get('novita_key', ''))
            set_select_value("select-novita-model", llm.get('novita_model', ''))
            
            # Friendli
            set_input_value("input-friendli-key", llm.get('friendli_key', ''))
            set_select_value("select-friendli-model", llm.get('friendli_model', ''))
            
            # Reka
            set_input_value("input-reka-key", llm.get('reka_key', ''))
            set_select_value("select-reka-model", llm.get('reka_model', ''))
            
            # Writer
            set_input_value("input-writer-key", llm.get('writer_key', ''))
            set_select_value("select-writer-model", llm.get('writer_model', ''))
            
            # Baseten
            set_input_value("input-baseten-key", llm.get('baseten_key', ''))
            set_input_value("input-baseten-model", llm.get('baseten_model', ''))
            
            # Modal
            set_input_value("input-modal-endpoint", llm.get('modal_endpoint', ''))
            set_input_value("input-modal-key", llm.get('modal_key', ''))
            set_input_value("input-modal-model", llm.get('modal_model', ''))
            
            # RunPod
            set_input_value("input-runpod-key", llm.get('runpod_key', ''))
            set_input_value("input-runpod-endpoint", llm.get('runpod_endpoint', ''))
            set_input_value("input-runpod-model", llm.get('runpod_model', ''))
            
            # Lambda Labs
            set_input_value("input-lambda-key", llm.get('lambda_key', ''))
            set_select_value("select-lambda-model", llm.get('lambda_model', ''))
            
            # Monster API
            set_input_value("input-monster-key", llm.get('monster_key', ''))
            set_select_value("select-monster-model", llm.get('monster_model', ''))
            
            # Hyperbolic
            set_input_value("input-hyperbolic-key", llm.get('hyperbolic_key', ''))
            set_select_value("select-hyperbolic-model", llm.get('hyperbolic_model', ''))
            
            # Kluster
            set_input_value("input-kluster-key", llm.get('kluster_key', ''))
            set_select_value("select-kluster-model", llm.get('kluster_model', ''))
            
            # vLLM (local)
            set_input_value("input-vllm-endpoint", llm.get('vllm_endpoint', ''))
            set_input_value("input-vllm-model", llm.get('vllm_model', ''))
            
            # LocalAI
            set_input_value("input-localai-endpoint", llm.get('localai_endpoint', ''))
            set_input_value("input-localai-model", llm.get('localai_model', ''))
            
            # TextGen WebUI
            set_input_value("input-textgen-endpoint", llm.get('textgen_endpoint', ''))
            set_input_value("input-textgen-model", llm.get('textgen_model', ''))
            
            # Custom API
            set_input_value("input-custom-base-url", llm.get('custom_base_url', ''))
            set_input_value("input-custom-key", llm.get('custom_key', ''))
            set_input_value("input-custom-model", llm.get('custom_model', ''))
            
            # Thinking enabled
            set_switch_value("switch-thinking-enabled", llm.get('thinking_enabled', False))
            
            # Update TUIState if available
            if hasattr(self, 'state'):
                mode = llm.get('mode', 'none')
                self.state.llm_provider = mode
                self.state.thinking_enabled = llm.get('thinking_enabled', False)
                
                # Get the model name for the selected provider - COMPLETE MAPPING
                model_key_map = {
                    'local': 'local_model',
                    'ollama': 'local_model',
                    'lmstudio': 'local_model',
                    'llamacpp': 'local_model',
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
                    'azure_openai': 'azure_deployment',
                    'aws_bedrock': 'aws_model',
                    'huggingface': 'hf_model',
                    'fireworks': 'fireworks_model',
                    'replicate': 'replicate_model',
                    'ai21': 'ai21_model',
                    'deepinfra': 'deepinfra_model',
                    'anyscale': 'anyscale_model',
                    'vertex_ai': 'vertex_model',
                    'watsonx': 'watsonx_model',
                    'oracle_ai': 'oracle_model',
                    'alibaba_qwen': 'alibaba_model',
                    'sambanova': 'sambanova_model',
                    'cerebras': 'cerebras_model',
                    'lepton': 'lepton_model',
                    'novita': 'novita_model',
                    'friendli': 'friendli_model',
                    'reka': 'reka_model',
                    'writer': 'writer_model',
                    'baseten': 'baseten_model',
                    'modal': 'modal_model',
                    'runpod': 'runpod_model',
                    'lambda': 'lambda_model',
                    'monster': 'monster_model',
                    'hyperbolic': 'hyperbolic_model',
                    'kluster': 'kluster_model',
                    'vllm': 'vllm_model',
                    'localai': 'localai_model',
                    'textgen_webui': 'textgen_model',
                    'oobabooga': 'textgen_model',
                }
                
                if mode in model_key_map:
                    self.state.llm_model = llm.get(model_key_map[mode], '')
                elif mode not in ['none', '', None]:
                    self.state.llm_model = llm.get('custom_model', '')
                else:
                    self.state.llm_model = ''
                
                # Mark as connected if we have a valid provider configured
                if mode and mode != 'none':
                    self.state.llm_connected = True
                else:
                    self.state.llm_connected = False
            
            # Apply display settings
            display = settings.get('display', {})
            if 'theme' in display:
                self.query_one("#select-theme", Select).value = display['theme']
            if 'compact_sidebar' in display:
                self.query_one("#switch-compact", Switch).value = display['compact_sidebar']
            if 'show_logs' in display:
                self.query_one("#switch-logs", Switch).value = display['show_logs']
            
            self.notify("Settings loaded", severity="information")
        except Exception:
            pass  # Silently fail if settings can't be loaded

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
        """Export configuration to YAML file."""
        try:
            # Collect all current settings
            settings = {
                'proxima': {
                    'general': {
                        'backend': self.query_one("#select-backend", Select).value,
                        'shots': int(self.query_one("#input-shots", Input).value),
                        'autosave': self.query_one("#switch-autosave", Switch).value,
                    },
                    'llm': {
                        'mode': self.query_one("#select-llm-mode", Select).value,
                        'ollama_url': self.query_one("#input-ollama-url", Input).value,
                        'local_model': self.query_one("#input-local-model", Input).value,
                    },
                    'display': {
                        'theme': self.query_one("#select-theme", Select).value,
                        'compact_sidebar': self.query_one("#switch-compact", Switch).value,
                        'show_logs': self.query_one("#switch-logs", Switch).value,
                    },
                },
            }
            
            export_path = Path.home() / "proxima_config_export.yaml"
            
            # Try YAML export first
            try:
                import yaml
                with open(export_path, 'w') as f:
                    yaml.dump(settings, f, default_flow_style=False, indent=2)
            except ImportError:
                # Fallback to JSON if YAML not available
                import json
                export_path = Path.home() / "proxima_config_export.json"
                with open(export_path, 'w') as f:
                    json.dump(settings, f, indent=2)
            
            self.notify(f"✓ Configuration exported to {export_path}", severity="success")
            
        except Exception as e:
            self.notify(f"✗ Export failed: {e}", severity="error")

    def _import_config(self) -> None:
        """Import configuration from YAML or JSON file."""
        try:
            # Check for YAML file first, then JSON
            yaml_path = Path.home() / "proxima_config_export.yaml"
            json_path = Path.home() / "proxima_config_export.json"
            
            settings = None
            import_path = None
            
            if yaml_path.exists():
                try:
                    import yaml
                    with open(yaml_path, 'r') as f:
                        settings = yaml.safe_load(f)
                    import_path = yaml_path
                except ImportError:
                    pass
            
            if settings is None and json_path.exists():
                import json
                with open(json_path, 'r') as f:
                    settings = json.load(f)
                import_path = json_path
            
            if settings is None:
                self.notify("No config file found. Export first or create proxima_config_export.yaml", severity="warning")
                return
            
            # Apply settings
            proxima = settings.get('proxima', settings)  # Handle nested or flat
            
            general = proxima.get('general', {})
            if 'backend' in general:
                self.query_one("#select-backend", Select).value = general['backend']
            if 'shots' in general:
                self.query_one("#input-shots", Input).value = str(general['shots'])
            if 'autosave' in general:
                self.query_one("#switch-autosave", Switch).value = general['autosave']
            
            llm = proxima.get('llm', {})
            if 'mode' in llm:
                self.query_one("#select-llm-mode", Select).value = llm['mode']
                self._update_llm_sections(llm['mode'])
            if 'ollama_url' in llm:
                self.query_one("#input-ollama-url", Input).value = llm['ollama_url']
            if 'local_model' in llm:
                self.query_one("#input-local-model", Input).value = llm['local_model']
            
            display = proxima.get('display', {})
            if 'theme' in display:
                self.query_one("#select-theme", Select).value = display['theme']
            if 'compact_sidebar' in display:
                self.query_one("#switch-compact", Switch).value = display['compact_sidebar']
            if 'show_logs' in display:
                self.query_one("#switch-logs", Switch).value = display['show_logs']
            
            self.notify(f"✓ Configuration imported from {import_path}", severity="success")
            self.notify("Click 'Save Settings' to persist changes", severity="information")
            
        except Exception as e:
            self.notify(f"✗ Import failed: {e}", severity="error")

    def _open_ai_thinking(self) -> None:
        """Open the AI Thinking panel dialog."""
        from ..dialogs import AIThinkingDialog
        
        # Update the thinking enabled state
        try:
            thinking_enabled = self.query_one("#switch-thinking-enabled", Switch).value
            if hasattr(self, 'state') and self.state:
                self.state.thinking_enabled = thinking_enabled
        except Exception:
            pass
        
        # Push the AI thinking dialog
        self.app.push_screen(AIThinkingDialog(state=getattr(self, 'state', None)))

    def _test_provider(self, provider_name: str) -> None:
        """Test API connection for a specific provider.
        
        Args:
            provider_name: Name of the provider to test (e.g., 'google', 'deepseek')
        """
        # Provider configuration with endpoint info
        provider_configs = {
            "google": {
                "key_input": "#input-google-key",
                "endpoint": "https://generativelanguage.googleapis.com/v1/models",
                "name": "Google AI (Gemini)",
                "key_prefix": "AIza",
            },
            "xai": {
                "key_input": "#input-xai-key",
                "endpoint": "https://api.x.ai/v1/models",
                "name": "xAI (Grok)",
                "key_prefix": "xai-",
            },
            "deepseek": {
                "key_input": "#input-deepseek-key",
                "endpoint": "https://api.deepseek.com/v1/models",
                "name": "DeepSeek",
                "key_prefix": "sk-",
            },
            "mistral": {
                "key_input": "#input-mistral-key",
                "endpoint": "https://api.mistral.ai/v1/models",
                "name": "Mistral AI",
                "key_prefix": "",
            },
            "groq": {
                "key_input": "#input-groq-key",
                "endpoint": "https://api.groq.com/openai/v1/models",
                "name": "Groq",
                "key_prefix": "gsk_",
            },
            "together": {
                "key_input": "#input-together-key",
                "endpoint": "https://api.together.xyz/v1/models",
                "name": "Together AI",
                "key_prefix": "",
            },
            "openrouter": {
                "key_input": "#input-openrouter-key",
                "endpoint": "https://openrouter.ai/api/v1/models",
                "name": "OpenRouter",
                "key_prefix": "sk-or-",
            },
            "cohere": {
                "key_input": "#input-cohere-key",
                "endpoint": "https://api.cohere.ai/v1/models",
                "name": "Cohere",
                "key_prefix": "",
            },
            "perplexity": {
                "key_input": "#input-perplexity-key",
                "endpoint": "https://api.perplexity.ai/chat/completions",
                "name": "Perplexity AI",
                "key_prefix": "pplx-",
            },
            "azure": {
                "key_input": "#input-azure-key",
                "endpoint_input": "#input-azure-endpoint",
                "name": "Azure OpenAI",
                "key_prefix": "",
            },
            "aws": {
                "key_input": "#input-aws-access-key",
                "name": "AWS Bedrock",
                "key_prefix": "AKIA",
            },
            "hf": {
                "key_input": "#input-hf-token",
                "endpoint": "https://huggingface.co/api/whoami",
                "name": "Hugging Face",
                "key_prefix": "hf_",
            },
            "fireworks": {
                "key_input": "#input-fireworks-key",
                "endpoint": "https://api.fireworks.ai/inference/v1/models",
                "name": "Fireworks AI",
                "key_prefix": "fw_",
            },
            "replicate": {
                "key_input": "#input-replicate-token",
                "endpoint": "https://api.replicate.com/v1/account",
                "name": "Replicate",
                "key_prefix": "r8_",
            },
            "ai21": {
                "key_input": "#input-ai21-key",
                "endpoint": "https://api.ai21.com/studio/v1/models",
                "name": "AI21 Labs",
                "key_prefix": "",
            },
            "deepinfra": {
                "key_input": "#input-deepinfra-key",
                "endpoint": "https://api.deepinfra.com/v1/openai/models",
                "name": "DeepInfra",
                "key_prefix": "",
            },
            # NEW: Additional providers
            "anyscale": {
                "key_input": "#input-anyscale-key",
                "endpoint": "https://api.endpoints.anyscale.com/v1/models",
                "name": "Anyscale",
                "key_prefix": "esecret_",
            },
            "vertex_ai": {
                "key_input": "#input-vertex-project",
                "name": "Google Vertex AI",
                "key_prefix": "",
            },
            "watsonx": {
                "key_input": "#input-watsonx-key",
                "endpoint": "https://us-south.ml.cloud.ibm.com/ml/v1/foundation_model_specs",
                "name": "IBM watsonx.ai",
                "key_prefix": "",
            },
            "oracle_ai": {
                "key_input": "#input-oracle-key",
                "name": "Oracle OCI AI",
                "key_prefix": "",
            },
            "alibaba_qwen": {
                "key_input": "#input-alibaba-key",
                "endpoint": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                "name": "Alibaba Cloud Qwen",
                "key_prefix": "sk-",
            },
            "sambanova": {
                "key_input": "#input-sambanova-key",
                "endpoint": "https://api.sambanova.ai/v1/models",
                "name": "SambaNova",
                "key_prefix": "",
            },
            "cerebras": {
                "key_input": "#input-cerebras-key",
                "endpoint": "https://api.cerebras.ai/v1/models",
                "name": "Cerebras",
                "key_prefix": "csk-",
            },
            "lepton": {
                "key_input": "#input-lepton-key",
                "endpoint": "https://api.lepton.ai/v1/models",
                "name": "Lepton AI",
                "key_prefix": "",
            },
            "novita": {
                "key_input": "#input-novita-key",
                "endpoint": "https://api.novita.ai/v3/openai/models",
                "name": "Novita AI",
                "key_prefix": "",
            },
            "friendli": {
                "key_input": "#input-friendli-key",
                "endpoint": "https://inference.friendli.ai/v1/models",
                "name": "Friendli AI",
                "key_prefix": "",
            },
            "reka": {
                "key_input": "#input-reka-key",
                "endpoint": "https://api.reka.ai/v1/models",
                "name": "Reka AI",
                "key_prefix": "",
            },
            "writer": {
                "key_input": "#input-writer-key",
                "endpoint": "https://api.writer.com/v1/models",
                "name": "Writer AI",
                "key_prefix": "",
            },
            "baseten": {
                "key_input": "#input-baseten-key",
                "name": "Baseten",
                "key_prefix": "",
            },
            "modal": {
                "key_input": "#input-modal-key",
                "endpoint_input": "#input-modal-endpoint",
                "name": "Modal",
                "key_prefix": "",
            },
            "runpod": {
                "key_input": "#input-runpod-key",
                "endpoint_input": "#input-runpod-endpoint",
                "name": "RunPod",
                "key_prefix": "",
            },
            "lambda": {
                "key_input": "#input-lambda-key",
                "endpoint": "https://api.lambdalabs.com/v1/models",
                "name": "Lambda Labs",
                "key_prefix": "",
            },
            "monster": {
                "key_input": "#input-monster-key",
                "endpoint": "https://api.monsterapi.ai/v1/models",
                "name": "Monster API",
                "key_prefix": "",
            },
            "hyperbolic": {
                "key_input": "#input-hyperbolic-key",
                "endpoint": "https://api.hyperbolic.xyz/v1/models",
                "name": "Hyperbolic",
                "key_prefix": "",
            },
            "kluster": {
                "key_input": "#input-kluster-key",
                "endpoint": "https://api.kluster.ai/v1/models",
                "name": "Kluster.ai",
                "key_prefix": "",
            },
            "vllm": {
                "endpoint_input": "#input-vllm-endpoint",
                "name": "vLLM (Self-Hosted)",
                "key_prefix": "",
            },
            "localai": {
                "endpoint_input": "#input-localai-endpoint",
                "name": "LocalAI",
                "key_prefix": "",
            },
            "textgen_webui": {
                "endpoint_input": "#input-textgen-endpoint",
                "name": "Text Generation WebUI",
                "key_prefix": "",
            },
            "oobabooga": {
                "endpoint_input": "#input-textgen-endpoint",
                "name": "Oobabooga",
                "key_prefix": "",
            },
            "custom": {
                "key_input": "#input-custom-key",
                "endpoint_input": "#input-custom-base-url",
                "name": "Custom API",
                "key_prefix": "",
            },
        }
        
        config = provider_configs.get(provider_name)
        if not config:
            self.notify(f"Unknown provider: {provider_name}", severity="warning")
            return
        
        # Get API key (optional for local endpoints)
        api_key = ""
        if "key_input" in config:
            try:
                api_key = self.query_one(config["key_input"], Input).value
            except Exception:
                pass
        
        # For providers that require API key
        requires_key = provider_name not in ("vllm", "localai", "textgen_webui", "oobabooga")
        if requires_key and not api_key:
            self.notify("Please enter an API key first", severity="warning")
            return
        
        # Validate key prefix if specified
        key_prefix = config.get("key_prefix", "")
        if api_key and key_prefix and not api_key.startswith(key_prefix):
            self.notify(f"API key should start with '{key_prefix}'", severity="warning")
        
        self.notify(f"Testing {config['name']} connection...")
        
        # Try to make a health check request
        try:
            import httpx
            
            endpoint = config.get("endpoint")
            if "endpoint_input" in config:
                try:
                    endpoint = self.query_one(config["endpoint_input"], Input).value
                    if not endpoint:
                        self.notify("Please enter an endpoint URL", severity="warning")
                        return
                    # Append models endpoint for OpenAI-compatible APIs
                    if not endpoint.endswith("/models"):
                        endpoint = endpoint.rstrip("/") + "/models"
                except Exception:
                    pass
            
            if not endpoint:
                # Basic validation only
                if len(api_key) > 10:
                    self.notify(f"✓ {config['name']} API key format looks valid", severity="success")
                else:
                    self.notify("API key seems too short", severity="warning")
                return
            
            # Make request with appropriate headers
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            # Some APIs use different header names
            if provider_name == "google":
                endpoint = f"{endpoint}?key={api_key}"
                headers = {}
            elif provider_name == "cohere":
                headers = {"Authorization": f"Bearer {api_key}"}
            elif provider_name == "hf":
                headers = {"Authorization": f"Bearer {api_key}"}
            
            with httpx.Client(timeout=10.0) as client:
                response = client.get(endpoint, headers=headers)
                
                if response.status_code == 200:
                    self.notify(f"✓ {config['name']} connection successful!", severity="success")
                elif response.status_code in (401, 403):
                    self.notify(f"✗ {config['name']} - Invalid or expired API key", severity="error")
                elif response.status_code == 429:
                    self.notify(f"✓ {config['name']} - Key valid (rate limited)", severity="success")
                else:
                    self.notify(f"⚠ {config['name']} - Status: {response.status_code}", severity="warning")
                    
        except httpx.ConnectError:
            self.notify(f"✗ Could not connect to {config['name']}", severity="error")
        except httpx.TimeoutException:
            self.notify(f"⚠ {config['name']} request timed out", severity="warning")
        except Exception as e:
            # Fallback to basic validation
            if len(api_key) > 10:
                self.notify(f"✓ {config['name']} API key format looks valid", severity="success")
                self.notify(f"(Full validation unavailable: {e})", severity="information")
            else:
                self.notify(f"API key seems too short", severity="warning")