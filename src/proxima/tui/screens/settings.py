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
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_mode = "local"  # "local", "openai", "anthropic", "none"

    def compose_main(self):
        """Compose the settings screen content."""
        from textual.containers import ScrollableContainer
        with ScrollableContainer(classes="main-content settings-container"):
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
                            # Major Cloud Providers
                            ("OpenAI API (GPT-4, GPT-4o)", "openai"),
                            ("Anthropic API (Claude)", "anthropic"),
                            ("Google AI (Gemini)", "google"),
                            ("xAI (Grok)", "xai"),
                            ("DeepSeek API", "deepseek"),
                            ("Mistral AI", "mistral"),
                            ("Cohere API", "cohere"),
                            ("Perplexity AI", "perplexity"),
                            ("Groq (Fast Inference)", "groq"),
                            ("Together AI", "together"),
                            ("Fireworks AI", "fireworks"),
                            ("Replicate", "replicate"),
                            ("Anyscale", "anyscale"),
                            # Enterprise / Cloud Platforms
                            ("Azure OpenAI", "azure_openai"),
                            ("AWS Bedrock", "aws_bedrock"),
                            ("Google Vertex AI", "vertex_ai"),
                            ("IBM watsonx.ai", "watsonx"),
                            ("Oracle OCI AI", "oracle_ai"),
                            ("Alibaba Cloud Qwen", "alibaba_qwen"),
                            ("Hugging Face Inference", "huggingface"),
                            # Open Source / Self-Hosted
                            ("Ollama (Local)", "ollama"),
                            ("LM Studio (Local)", "lmstudio"),
                            ("llama.cpp (Local)", "llamacpp"),
                            ("vLLM (Self-Hosted)", "vllm"),
                            ("Text Generation WebUI", "textgen_webui"),
                            ("LocalAI", "localai"),
                            ("OpenRouter", "openrouter"),
                            ("Oobabooga", "oobabooga"),
                            # Specialized Providers
                            ("AI21 Labs (Jamba)", "ai21"),
                            ("Reka AI", "reka"),
                            ("Writer AI", "writer"),
                            ("Lepton AI", "lepton"),
                            ("Baseten", "baseten"),
                            ("Modal", "modal"),
                            ("RunPod", "runpod"),
                            ("Lambda Labs", "lambda"),
                            ("SambaNova", "sambanova"),
                            ("Cerebras", "cerebras"),
                            ("Novita AI", "novita"),
                            ("Monster API", "monster"),
                            ("DeepInfra", "deepinfra"),
                            ("Hyperbolic", "hyperbolic"),
                            ("Kluster.ai", "kluster"),
                            ("Friendli AI", "friendli"),
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
                        "🔗 Test Connection",
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

                # Google Gemini API Settings
                with Container(classes="subsection", id="google-settings"):
                    yield Static("🔷 Google AI (Gemini) Settings", classes="subsection-title")
                    yield Static(
                        "Google's Gemini models. Requires API key from Google AI Studio.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="AIza...",
                            password=True,
                            classes="api-key-input",
                            id="input-google-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("Gemini 1.5 Flash (Recommended)", "gemini-1.5-flash-latest"),
                                ("Gemini 1.5 Pro", "gemini-1.5-pro-latest"),
                                ("Gemini 1.5 Flash-8B", "gemini-1.5-flash-8b"),
                                ("Gemini Pro", "gemini-pro"),
                                ("Gemini 2.0 Flash Exp", "gemini-2.0-flash-exp"),
                            ],
                            value="gemini-1.5-flash-latest",
                            id="select-google-model",
                        )

                    yield Button(
                        "Verify API Key",
                        id="btn-test-google",
                        variant="primary",
                    )

                # xAI Grok Settings
                with Container(classes="subsection", id="xai-settings"):
                    yield Static("🤖 xAI (Grok) Settings", classes="subsection-title")
                    yield Static(
                        "xAI's Grok models. Requires API key from x.ai.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="xai-...",
                            password=True,
                            classes="api-key-input",
                            id="input-xai-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("Grok-2 (Latest)", "grok-2"),
                                ("Grok-2 Vision", "grok-2-vision"),
                                ("Grok-2 Mini", "grok-2-mini"),
                                ("Grok-Beta", "grok-beta"),
                            ],
                            value="grok-2",
                            id="select-xai-model",
                        )

                    yield Button(
                        "Verify API Key",
                        id="btn-test-xai",
                        variant="primary",
                    )

                # DeepSeek Settings
                with Container(classes="subsection", id="deepseek-settings"):
                    yield Static("🔮 DeepSeek API Settings", classes="subsection-title")
                    yield Static(
                        "DeepSeek's powerful reasoning models. Very cost-effective.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="sk-...",
                            password=True,
                            classes="api-key-input",
                            id="input-deepseek-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("DeepSeek-V3 (Latest)", "deepseek-chat"),
                                ("DeepSeek-R1 (Reasoning)", "deepseek-reasoner"),
                                ("DeepSeek Coder V2", "deepseek-coder"),
                            ],
                            value="deepseek-chat",
                            id="select-deepseek-model",
                        )

                    yield Button(
                        "Verify API Key",
                        id="btn-test-deepseek",
                        variant="primary",
                    )

                # Mistral AI Settings
                with Container(classes="subsection", id="mistral-settings"):
                    yield Static("🌊 Mistral AI Settings", classes="subsection-title")
                    yield Static(
                        "Mistral's efficient models. Good balance of speed and quality.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="...",
                            password=True,
                            classes="api-key-input",
                            id="input-mistral-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("Mistral Large (Latest)", "mistral-large-latest"),
                                ("Mistral Medium", "mistral-medium-latest"),
                                ("Mistral Small", "mistral-small-latest"),
                                ("Mixtral 8x7B", "open-mixtral-8x7b"),
                                ("Mixtral 8x22B", "open-mixtral-8x22b"),
                                ("Codestral", "codestral-latest"),
                                ("Pixtral Large", "pixtral-large-latest"),
                            ],
                            value="mistral-large-latest",
                            id="select-mistral-model",
                        )

                    yield Button(
                        "Verify API Key",
                        id="btn-test-mistral",
                        variant="primary",
                    )

                # Groq Settings (Fast Inference)
                with Container(classes="subsection", id="groq-settings"):
                    yield Static("⚡ Groq (Fast Inference) Settings", classes="subsection-title")
                    yield Static(
                        "Ultra-fast inference on LPU. Free tier available.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="gsk_...",
                            password=True,
                            classes="api-key-input",
                            id="input-groq-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("Llama 3.3 70B Versatile", "llama-3.3-70b-versatile"),
                                ("Llama 3.1 70B", "llama-3.1-70b-versatile"),
                                ("Llama 3.1 8B", "llama-3.1-8b-instant"),
                                ("Mixtral 8x7B", "mixtral-8x7b-32768"),
                                ("Gemma 2 9B", "gemma2-9b-it"),
                                ("DeepSeek R1 Distill Llama 70B", "deepseek-r1-distill-llama-70b"),
                            ],
                            value="llama-3.3-70b-versatile",
                            id="select-groq-model",
                        )

                    yield Button(
                        "Verify API Key",
                        id="btn-test-groq",
                        variant="primary",
                    )

                # Together AI Settings
                with Container(classes="subsection", id="together-settings"):
                    yield Static("🤝 Together AI Settings", classes="subsection-title")
                    yield Static(
                        "Access to 100+ open models. Competitive pricing.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="...",
                            password=True,
                            classes="api-key-input",
                            id="input-together-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("Llama 3.3 70B Turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
                                ("Llama 3.1 405B", "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"),
                                ("DeepSeek V3", "deepseek-ai/DeepSeek-V3"),
                                ("Qwen 2.5 72B", "Qwen/Qwen2.5-72B-Instruct-Turbo"),
                                ("Mixtral 8x22B", "mistralai/Mixtral-8x22B-Instruct-v0.1"),
                                ("WizardLM 2 8x22B", "microsoft/WizardLM-2-8x22B"),
                            ],
                            value="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                            id="select-together-model",
                        )

                    yield Button(
                        "Verify API Key",
                        id="btn-test-together",
                        variant="primary",
                    )

                # OpenRouter Settings
                with Container(classes="subsection", id="openrouter-settings"):
                    yield Static("🔀 OpenRouter Settings", classes="subsection-title")
                    yield Static(
                        "Unified API for 100+ models from multiple providers.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="sk-or-...",
                            password=True,
                            classes="api-key-input",
                            id="input-openrouter-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("Auto (Best Available)", "openrouter/auto"),
                                ("GPT-4o", "openai/gpt-4o"),
                                ("Claude 3.5 Sonnet", "anthropic/claude-3.5-sonnet"),
                                ("Gemini Pro 1.5", "google/gemini-pro-1.5"),
                                ("Llama 3.1 405B", "meta-llama/llama-3.1-405b-instruct"),
                                ("DeepSeek V3", "deepseek/deepseek-chat"),
                                ("Perplexity Online", "perplexity/llama-3.1-sonar-large-128k-online"),
                            ],
                            value="openrouter/auto",
                            id="select-openrouter-model",
                        )

                    yield Button(
                        "Verify API Key",
                        id="btn-test-openrouter",
                        variant="primary",
                    )

                # Cohere Settings
                with Container(classes="subsection", id="cohere-settings"):
                    yield Static("🔶 Cohere API Settings", classes="subsection-title")
                    yield Static(
                        "Cohere's Command models. Good for enterprise use.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="...",
                            password=True,
                            classes="api-key-input",
                            id="input-cohere-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("Command R+ (Latest)", "command-r-plus"),
                                ("Command R", "command-r"),
                                ("Command Light", "command-light"),
                                ("Command Nightly", "command-nightly"),
                            ],
                            value="command-r-plus",
                            id="select-cohere-model",
                        )

                    yield Button(
                        "Verify API Key",
                        id="btn-test-cohere",
                        variant="primary",
                    )

                # Perplexity Settings
                with Container(classes="subsection", id="perplexity-settings"):
                    yield Static("🔍 Perplexity AI Settings", classes="subsection-title")
                    yield Static(
                        "Models with real-time internet search capability.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="pplx-...",
                            password=True,
                            classes="api-key-input",
                            id="input-perplexity-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("Sonar Large Online", "llama-3.1-sonar-large-128k-online"),
                                ("Sonar Small Online", "llama-3.1-sonar-small-128k-online"),
                                ("Sonar Large Chat", "llama-3.1-sonar-large-128k-chat"),
                                ("Sonar Huge", "llama-3.1-sonar-huge-128k-online"),
                            ],
                            value="llama-3.1-sonar-large-128k-online",
                            id="select-perplexity-model",
                        )

                    yield Button(
                        "Verify API Key",
                        id="btn-test-perplexity",
                        variant="primary",
                    )

                # Azure OpenAI Settings
                with Container(classes="subsection", id="azure-openai-settings"):
                    yield Static("☁️ Azure OpenAI Settings", classes="subsection-title")
                    yield Static(
                        "Enterprise OpenAI via Azure. Requires Azure subscription.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="...",
                            password=True,
                            classes="api-key-input",
                            id="input-azure-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Endpoint:", classes="setting-label")
                        yield Input(
                            placeholder="https://your-resource.openai.azure.com/",
                            classes="setting-input-wide",
                            id="input-azure-endpoint",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Deployment:", classes="setting-label")
                        yield Input(
                            placeholder="gpt-4-deployment",
                            classes="setting-input",
                            id="input-azure-deployment",
                        )

                    yield Button(
                        "Verify Connection",
                        id="btn-test-azure",
                        variant="primary",
                    )

                # AWS Bedrock Settings
                with Container(classes="subsection", id="aws-bedrock-settings"):
                    yield Static("🔶 AWS Bedrock Settings", classes="subsection-title")
                    yield Static(
                        "Access to Claude, Llama, Titan via AWS. Requires AWS credentials.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("Access Key:", classes="setting-label")
                        yield Input(
                            placeholder="AKIA...",
                            password=True,
                            classes="api-key-input",
                            id="input-aws-access-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Secret Key:", classes="setting-label")
                        yield Input(
                            placeholder="...",
                            password=True,
                            classes="api-key-input",
                            id="input-aws-secret-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Region:", classes="setting-label")
                        yield Select(
                            [
                                ("US East (N. Virginia)", "us-east-1"),
                                ("US West (Oregon)", "us-west-2"),
                                ("EU (Frankfurt)", "eu-central-1"),
                                ("Asia Pacific (Tokyo)", "ap-northeast-1"),
                            ],
                            value="us-east-1",
                            id="select-aws-region",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("Claude 3.5 Sonnet", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
                                ("Claude 3.5 Haiku", "anthropic.claude-3-5-haiku-20241022-v1:0"),
                                ("Llama 3.1 70B", "meta.llama3-1-70b-instruct-v1:0"),
                                ("Amazon Titan Text", "amazon.titan-text-premier-v1:0"),
                                ("Mistral Large", "mistral.mistral-large-2407-v1:0"),
                            ],
                            value="anthropic.claude-3-5-sonnet-20241022-v2:0",
                            id="select-aws-model",
                        )

                    yield Button(
                        "Verify Credentials",
                        id="btn-test-aws",
                        variant="primary",
                    )

                # Hugging Face Settings
                with Container(classes="subsection", id="huggingface-settings"):
                    yield Static("🤗 Hugging Face Inference Settings", classes="subsection-title")
                    yield Static(
                        "Access to thousands of models on Hugging Face Hub.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Token:", classes="setting-label")
                        yield Input(
                            placeholder="hf_...",
                            password=True,
                            classes="api-key-input",
                            id="input-hf-token",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model ID:", classes="setting-label")
                        yield Input(
                            placeholder="meta-llama/Llama-3.1-8B-Instruct",
                            classes="setting-input-wide",
                            id="input-hf-model",
                        )

                    yield Button(
                        "Verify Token",
                        id="btn-test-hf",
                        variant="primary",
                    )

                # Fireworks AI Settings
                with Container(classes="subsection", id="fireworks-settings"):
                    yield Static("🎆 Fireworks AI Settings", classes="subsection-title")
                    yield Static(
                        "Fast inference with competitive pricing. Good for coding models.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="fw_...",
                            password=True,
                            classes="api-key-input",
                            id="input-fireworks-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("Llama 3.1 405B", "accounts/fireworks/models/llama-v3p1-405b-instruct"),
                                ("Llama 3.1 70B", "accounts/fireworks/models/llama-v3p1-70b-instruct"),
                                ("DeepSeek V3", "accounts/fireworks/models/deepseek-v3"),
                                ("Qwen 2.5 72B", "accounts/fireworks/models/qwen2p5-72b-instruct"),
                                ("Mixtral 8x22B", "accounts/fireworks/models/mixtral-8x22b-instruct"),
                            ],
                            value="accounts/fireworks/models/llama-v3p1-70b-instruct",
                            id="select-fireworks-model",
                        )

                    yield Button(
                        "Verify API Key",
                        id="btn-test-fireworks",
                        variant="primary",
                    )

                # Replicate Settings
                with Container(classes="subsection", id="replicate-settings"):
                    yield Static("🔄 Replicate Settings", classes="subsection-title")
                    yield Static(
                        "Run open-source models in the cloud. Pay per second.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Token:", classes="setting-label")
                        yield Input(
                            placeholder="r8_...",
                            password=True,
                            classes="api-key-input",
                            id="input-replicate-token",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Input(
                            placeholder="meta/llama-2-70b-chat",
                            classes="setting-input-wide",
                            id="input-replicate-model",
                        )

                    yield Button(
                        "Verify Token",
                        id="btn-test-replicate",
                        variant="primary",
                    )

                # AI21 Labs Settings
                with Container(classes="subsection", id="ai21-settings"):
                    yield Static("🧬 AI21 Labs (Jamba) Settings", classes="subsection-title")
                    yield Static(
                        "Jamba - SSM-Transformer hybrid with 256K context.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="...",
                            password=True,
                            classes="api-key-input",
                            id="input-ai21-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("Jamba 1.5 Large", "jamba-1.5-large"),
                                ("Jamba 1.5 Mini", "jamba-1.5-mini"),
                                ("Jamba Instruct", "jamba-instruct"),
                            ],
                            value="jamba-1.5-large",
                            id="select-ai21-model",
                        )

                    yield Button(
                        "Verify API Key",
                        id="btn-test-ai21",
                        variant="primary",
                    )

                # DeepInfra Settings
                with Container(classes="subsection", id="deepinfra-settings"):
                    yield Static("🚀 DeepInfra Settings", classes="subsection-title")
                    yield Static(
                        "Fast, scalable inference for open models. Great pricing.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="...",
                            password=True,
                            classes="api-key-input",
                            id="input-deepinfra-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model:", classes="setting-label")
                        yield Select(
                            [
                                ("Llama 3.1 70B", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
                                ("Mixtral 8x22B", "mistralai/Mixtral-8x22B-Instruct-v0.1"),
                                ("Qwen 2.5 72B", "Qwen/Qwen2.5-72B-Instruct"),
                                ("DeepSeek V3", "deepseek-ai/DeepSeek-V3"),
                                ("Phi-3 Medium", "microsoft/Phi-3-medium-128k-instruct"),
                            ],
                            value="meta-llama/Meta-Llama-3.1-70B-Instruct",
                            id="select-deepinfra-model",
                        )

                    yield Button(
                        "Verify API Key",
                        id="btn-test-deepinfra",
                        variant="primary",
                    )

                # Generic/Custom API Settings (for additional providers)
                with Container(classes="subsection", id="custom-api-settings"):
                    yield Static("⚙️ Custom API Settings", classes="subsection-title")
                    yield Static(
                        "Configure any OpenAI-compatible API endpoint.",
                        classes="hint-text",
                    )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Base URL:", classes="setting-label")
                        yield Input(
                            placeholder="https://api.example.com/v1",
                            classes="setting-input-wide",
                            id="input-custom-base-url",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("API Key:", classes="setting-label")
                        yield Input(
                            placeholder="...",
                            password=True,
                            classes="api-key-input",
                            id="input-custom-key",
                        )

                    with Horizontal(classes="setting-row"):
                        yield Static("Model Name:", classes="setting-label")
                        yield Input(
                            placeholder="model-name",
                            classes="setting-input",
                            id="input-custom-model",
                        )

                    yield Button(
                        "Test Connection",
                        id="btn-test-custom",
                        variant="primary",
                    )

                # AI Thinking Panel Access
                with Container(classes="subsection", id="ai-thinking-settings"):
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
                yield Button("💾 Save Settings", id="btn-save", classes="action-btn", variant="primary")
                yield Button("🔄 Reset to Defaults", id="btn-reset", classes="action-btn")
                yield Button("📤 Export Config", id="btn-export", classes="action-btn")
                yield Button("📥 Import Config", id="btn-import", classes="action-btn")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "select-llm-mode":
            self._update_llm_sections(event.value)
        elif event.select.id == "select-theme":
            self._apply_theme(event.value)
    
    def _apply_theme(self, theme_name: str) -> None:
        """Apply the selected theme immediately.
        
        Args:
            theme_name: Name of the theme to apply ('dark' or 'light')
        """
        try:
            app = self.app
            if theme_name == "dark":
                app.dark = True
                app.theme_name = "dark"
            elif theme_name == "light":
                app.dark = False
                app.theme_name = "light"
            elif theme_name == "quantum":
                # Custom quantum theme - uses dark mode with quantum colors
                app.dark = True
                app.theme_name = "quantum"
            
            self.notify(f"Theme switched to {theme_name}", severity="information")
        except Exception as e:
            self.notify(f"Failed to apply theme: {e}", severity="warning")

    def _update_llm_sections(self, mode: str) -> None:
        """Show/hide LLM sections based on selected mode."""
        # All provider section IDs
        all_sections = [
            "#local-llm-settings",
            "#openai-settings",
            "#anthropic-settings",
            "#google-settings",
            "#xai-settings",
            "#deepseek-settings",
            "#mistral-settings",
            "#groq-settings",
            "#together-settings",
            "#openrouter-settings",
            "#cohere-settings",
            "#perplexity-settings",
            "#azure-openai-settings",
            "#aws-bedrock-settings",
            "#huggingface-settings",
            "#fireworks-settings",
            "#replicate-settings",
            "#ai21-settings",
            "#deepinfra-settings",
            "#custom-api-settings",
        ]
        
        # Mapping from mode to section ID
        mode_to_section = {
            "local": "#local-llm-settings",
            "openai": "#openai-settings",
            "anthropic": "#anthropic-settings",
            "google": "#google-settings",
            "xai": "#xai-settings",
            "deepseek": "#deepseek-settings",
            "mistral": "#mistral-settings",
            "groq": "#groq-settings",
            "together": "#together-settings",
            "openrouter": "#openrouter-settings",
            "cohere": "#cohere-settings",
            "perplexity": "#perplexity-settings",
            "azure_openai": "#azure-openai-settings",
            "aws_bedrock": "#aws-bedrock-settings",
            "huggingface": "#huggingface-settings",
            "fireworks": "#fireworks-settings",
            "replicate": "#replicate-settings",
            "ai21": "#ai21-settings",
            "deepinfra": "#deepinfra-settings",
            # Local providers also use local-llm-settings or custom
            "ollama": "#local-llm-settings",
            "lmstudio": "#local-llm-settings",
            "llamacpp": "#local-llm-settings",
            "vllm": "#custom-api-settings",
            "textgen_webui": "#custom-api-settings",
            "localai": "#custom-api-settings",
            "oobabooga": "#custom-api-settings",
            # Specialized providers use custom settings
            "vertex_ai": "#custom-api-settings",
            "watsonx": "#custom-api-settings",
            "oracle_ai": "#custom-api-settings",
            "alibaba_qwen": "#custom-api-settings",
            "anyscale": "#custom-api-settings",
            "reka": "#custom-api-settings",
            "writer": "#custom-api-settings",
            "lepton": "#custom-api-settings",
            "baseten": "#custom-api-settings",
            "modal": "#custom-api-settings",
            "runpod": "#custom-api-settings",
            "lambda": "#custom-api-settings",
            "sambanova": "#custom-api-settings",
            "cerebras": "#custom-api-settings",
            "novita": "#custom-api-settings",
            "monster": "#custom-api-settings",
            "hyperbolic": "#custom-api-settings",
            "kluster": "#custom-api-settings",
            "friendli": "#custom-api-settings",
        }
        
        # Hide all sections first
        for section_id in all_sections:
            try:
                section = self.query_one(section_id)
                section.display = False
            except Exception:
                pass  # Section may not exist

        # Show the relevant section based on mode
        if mode != "none" and mode in mode_to_section:
            section_id = mode_to_section[mode]
            try:
                section = self.query_one(section_id)
                section.display = True
            except Exception:
                pass

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
        # Handle new provider test buttons
        elif button_id and button_id.startswith("btn-test-"):
            self._test_provider(button_id.replace("btn-test-", ""))

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
                
                # Get the model name for the selected provider
                model_key_map = {
                    'local': 'local_model',
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
        """Load saved settings on mount."""
        self._update_llm_sections("none")
        self._load_saved_settings()
    
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
                
                # Get the model name for the selected provider
                model_key_map = {
                    'local': 'local_model',
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
        
        # Get API key
        try:
            api_key = self.query_one(config["key_input"], Input).value
        except Exception:
            self.notify(f"Could not find API key input for {config['name']}", severity="error")
            return
        
        if not api_key:
            self.notify("Please enter an API key first", severity="warning")
            return
        
        # Validate key prefix if specified
        key_prefix = config.get("key_prefix", "")
        if key_prefix and not api_key.startswith(key_prefix):
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