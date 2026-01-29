"""AI Assistant Screen for Proxima TUI.

Full-featured AI chat interface with:
- Persistent conversation memory
- Import/export chat functionality
- Keyboard shortcuts
- Real-time stats tracking
- Multi-line input support
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict

from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.widgets import Static, Button, Input, RichLog, TextArea
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown

from .base import BaseScreen
from ..styles.theme import get_theme

# Import LLM components
try:
    from proxima.intelligence.llm_router import LLMRouter, LLMRequest
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tokens: int = 0
    thinking_time_ms: int = 0


@dataclass
class ChatSession:
    """A complete chat session with metadata."""
    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    name: str = "New Chat"
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_tokens: int = 0
    total_requests: int = 0
    provider: str = ""
    model: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'messages': [asdict(m) for m in self.messages],
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'total_tokens': self.total_tokens,
            'total_requests': self.total_requests,
            'provider': self.provider,
            'model': self.model,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatSession':
        """Create from dictionary."""
        session = cls(
            id=data.get('id', datetime.now().strftime("%Y%m%d_%H%M%S")),
            name=data.get('name', 'Imported Chat'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            total_tokens=data.get('total_tokens', 0),
            total_requests=data.get('total_requests', 0),
            provider=data.get('provider', ''),
            model=data.get('model', ''),
        )
        session.messages = [
            ChatMessage(**m) for m in data.get('messages', [])
        ]
        return session


class AIAssistantScreen(BaseScreen):
    """Full-featured AI Assistant screen.
    
    Features:
    - Persistent chat history
    - Multi-line input (Shift+Enter for new line)
    - Import/export conversations
    - Real-time statistics
    - Keyboard shortcuts
    """
    
    SCREEN_NAME = "ai_assistant"
    SCREEN_TITLE = "AI Assistant"
    SHOW_SIDEBAR = False  # Full screen mode
    
    BINDINGS = [
        Binding("ctrl+z", "undo", "Undo", show=True),
        Binding("ctrl+y", "redo", "Redo", show=True),
        Binding("ctrl+a", "select_all", "Select All", show=False),
        Binding("ctrl+x", "cut", "Cut", show=False),
        Binding("ctrl+c", "copy_or_cancel", "Copy/Cancel", show=False),
        Binding("ctrl+j", "prev_prompt", "Previous Prompt", show=True),
        Binding("ctrl+l", "next_prompt", "Next Prompt", show=True),
        Binding("ctrl+n", "new_chat", "New Chat", show=True),
        Binding("ctrl+s", "export_chat", "Export Chat", show=True),
        Binding("ctrl+o", "import_chat", "Import Chat", show=True),
        Binding("ctrl+shift+c", "clear_chat", "Clear Chat", show=True),
        Binding("escape", "go_back", "Back"),
        Binding("f1", "show_shortcuts", "Show Shortcuts"),
        # Navigation
        ("1", "goto_dashboard", "Dashboard"),
        ("2", "goto_execution", "Execution"),
        ("3", "goto_results", "Results"),
        ("4", "goto_backends", "Backends"),
        ("5", "goto_settings", "Settings"),
        ("6", "goto_ai_assistant", "AI Assistant"),
    ]
    
    DEFAULT_CSS = """
    AIAssistantScreen {
        layout: horizontal;
    }
    
    AIAssistantScreen .main-container {
        width: 100%;
        height: 100%;
        layout: horizontal;
    }
    
    AIAssistantScreen .chat-area {
        width: 80%;
        height: 100%;
        border-right: solid $primary;
    }
    
    AIAssistantScreen .sidebar-panel {
        width: 20%;
        height: 100%;
        background: $surface-darken-1;
    }
    
    AIAssistantScreen .header-section {
        height: 5;
        padding: 1;
        background: $primary-darken-2;
        border-bottom: solid $primary;
    }
    
    AIAssistantScreen .header-title {
        text-style: bold;
        color: $accent;
        text-align: center;
    }
    
    AIAssistantScreen .header-subtitle {
        color: $text-muted;
        text-align: center;
    }
    
    AIAssistantScreen .chat-log-container {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    
    AIAssistantScreen .chat-log {
        height: 100%;
        background: $surface-darken-2;
        padding: 1;
        border: solid $primary-darken-3;
    }
    
    AIAssistantScreen .input-section {
        height: auto;
        min-height: 8;
        max-height: 15;
        padding: 1;
        border-top: solid $primary;
        background: $surface;
    }
    
    AIAssistantScreen .input-container {
        height: auto;
        min-height: 3;
        layout: horizontal;
    }
    
    AIAssistantScreen .prompt-input {
        width: 1fr;
        min-height: 3;
        max-height: 10;
        margin-right: 1;
    }
    
    AIAssistantScreen .send-btn {
        width: 12;
        height: 3;
    }
    
    AIAssistantScreen .controls-row {
        height: 3;
        layout: horizontal;
        margin-top: 1;
    }
    
    AIAssistantScreen .control-btn {
        margin-right: 1;
        min-width: 12;
        height: 3;
    }
    
    AIAssistantScreen .input-hint {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    
    /* Sidebar styles */
    AIAssistantScreen .sidebar-section {
        height: auto;
        padding: 1;
        border-bottom: solid $primary-darken-3;
    }
    
    AIAssistantScreen .sidebar-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    AIAssistantScreen .stats-grid {
        height: auto;
    }
    
    AIAssistantScreen .stat-row {
        height: auto;
        layout: horizontal;
        margin-bottom: 0;
    }
    
    AIAssistantScreen .stat-label {
        width: 12;
        color: $text-muted;
    }
    
    AIAssistantScreen .stat-value {
        width: 1fr;
        color: $text;
        text-align: right;
    }
    
    AIAssistantScreen .stat-highlight {
        color: $accent;
    }
    
    AIAssistantScreen .history-list {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    
    AIAssistantScreen .history-item {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        background: $surface-darken-2;
        border: solid $primary-darken-3;
    }
    
    AIAssistantScreen .history-item:hover {
        background: $primary-darken-2;
        border: solid $primary;
    }
    
    AIAssistantScreen .shortcuts-panel {
        height: auto;
        padding: 1;
    }
    
    AIAssistantScreen .shortcut-row {
        height: auto;
        layout: horizontal;
    }
    
    AIAssistantScreen .shortcut-key {
        width: 10;
        color: $accent;
    }
    
    AIAssistantScreen .shortcut-desc {
        width: 1fr;
        color: $text-muted;
    }
    """
    
    def __init__(self, **kwargs):
        """Initialize the AI Assistant screen."""
        super().__init__(**kwargs)
        
        # Chat state
        self._current_session: ChatSession = ChatSession()
        self._sessions: List[ChatSession] = []
        self._prompt_history: List[str] = []
        self._prompt_history_index: int = -1
        self._undo_stack: List[str] = []
        self._redo_stack: List[str] = []
        
        # LLM state
        self._llm_router: Optional[LLMRouter] = None
        self._llm_provider: str = 'none'
        self._llm_model: str = ''
        self._is_generating: bool = False
        
        # Stats
        self._stats = {
            'total_messages': 0,
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_requests': 0,
            'avg_response_time': 0,
            'session_start': time.time(),
        }
        
        # Initialize LLM router
        if LLM_AVAILABLE:
            try:
                def auto_consent(prompt: str) -> bool:
                    return True
                self._llm_router = LLMRouter(consent_prompt=auto_consent)
            except Exception:
                pass
        
        # Load settings and history
        self._load_settings()
        self._load_chat_history()
    
    def _load_settings(self) -> None:
        """Load LLM settings from config."""
        try:
            config_path = Path.home() / ".proxima" / "tui_settings.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    settings = json.load(f)
                
                llm = settings.get('llm', {})
                self._llm_provider = llm.get('mode', 'none')
                
                # Get model based on provider
                model_keys = {
                    'local': 'local_model', 'ollama': 'local_model',
                    'openai': 'openai_model', 'anthropic': 'anthropic_model',
                    'google': 'google_model', 'xai': 'xai_model',
                    'deepseek': 'deepseek_model', 'mistral': 'mistral_model',
                    'groq': 'groq_model', 'together': 'together_model',
                    'perplexity': 'perplexity_model', 'fireworks': 'fireworks_model',
                    'cohere': 'cohere_model', 'openrouter': 'openrouter_model',
                    'huggingface': 'huggingface_model', 'replicate': 'replicate_model',
                }
                self._llm_model = llm.get(model_keys.get(self._llm_provider, ''), '')
                
                # Get and register API key with the router
                api_key_map = {
                    'openai': 'openai_key', 'anthropic': 'anthropic_key',
                    'google': 'google_key', 'gemini': 'google_key',
                    'xai': 'xai_key', 'grok': 'xai_key',
                    'deepseek': 'deepseek_key', 'mistral': 'mistral_key',
                    'groq': 'groq_key', 'together': 'together_key',
                    'cohere': 'cohere_key', 'perplexity': 'perplexity_key',
                    'fireworks': 'fireworks_key', 'huggingface': 'huggingface_key',
                    'replicate': 'replicate_key', 'openrouter': 'openrouter_key',
                    'azure_openai': 'azure_openai_key', 'azure': 'azure_openai_key',
                }
                api_key_field = api_key_map.get(self._llm_provider)
                api_key = llm.get(api_key_field, '') if api_key_field else ''
                
                # Map provider to router name for API key registration
                provider_map = {
                    'local': 'ollama', 'ollama': 'ollama',
                    'openai': 'openai', 'anthropic': 'anthropic',
                    'google': 'google', 'gemini': 'google',
                    'xai': 'xai', 'grok': 'xai',
                    'deepseek': 'deepseek', 'mistral': 'mistral',
                    'groq': 'groq', 'together': 'together',
                    'openrouter': 'openrouter', 'cohere': 'cohere',
                    'lmstudio': 'lmstudio', 'llamacpp': 'llama_cpp',
                    'perplexity': 'perplexity', 'fireworks': 'fireworks',
                    'huggingface': 'huggingface', 'replicate': 'replicate',
                    'azure_openai': 'azure_openai', 'azure': 'azure_openai',
                }
                router_provider = provider_map.get(self._llm_provider, self._llm_provider)
                
                # Register API key with the router for proper authentication
                if api_key and self._llm_router and router_provider:
                    try:
                        self._llm_router.api_keys.store_key(router_provider, api_key)
                    except Exception:
                        pass  # Silently ignore if API key storage fails
                
                # Update session info
                self._current_session.provider = self._llm_provider
                self._current_session.model = self._llm_model
        except Exception:
            pass
    
    def _load_chat_history(self) -> None:
        """Load chat history from disk."""
        try:
            history_dir = Path.home() / ".proxima" / "chat_history"
            if not history_dir.exists():
                return
            
            for file_path in sorted(history_dir.glob("*.json"), reverse=True)[:10]:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    session = ChatSession.from_dict(data)
                    self._sessions.append(session)
                except Exception:
                    continue
        except Exception:
            pass
    
    def _save_current_session(self) -> None:
        """Save current session to disk."""
        try:
            history_dir = Path.home() / ".proxima" / "chat_history"
            history_dir.mkdir(parents=True, exist_ok=True)
            
            self._current_session.updated_at = datetime.now().isoformat()
            file_path = history_dir / f"chat_{self._current_session.id}.json"
            
            with open(file_path, 'w') as f:
                json.dump(self._current_session.to_dict(), f, indent=2)
        except Exception:
            pass
    
    def on_mount(self) -> None:
        """Called when screen is mounted."""
        self._update_stats_display()
        self._show_welcome_message()
        
        # Focus the input
        self.set_timer(0.1, self._focus_input)
    
    def _focus_input(self) -> None:
        """Focus the prompt input."""
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
            input_widget.focus()
        except Exception:
            pass
    
    def _show_welcome_message(self) -> None:
        """Show welcome message in chat log."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", RichLog)
            
            welcome = Text()
            welcome.append("🤖 AI Assistant\n", style=f"bold {theme.accent}")
            welcome.append("━" * 50 + "\n\n", style=theme.border)
            
            welcome.append("Welcome! I'm your AI assistant for Proxima.\n\n", style=theme.fg_base)
            welcome.append("I can help you with:\n", style=theme.fg_muted)
            welcome.append("• Quantum circuit design and optimization\n", style=theme.fg_base)
            welcome.append("• Backend configuration and selection\n", style=theme.fg_base)
            welcome.append("• Performance analysis and troubleshooting\n", style=theme.fg_base)
            welcome.append("• Understanding quantum computing concepts\n\n", style=theme.fg_base)
            
            if self._llm_provider and self._llm_provider != 'none':
                welcome.append(f"Provider: ", style=theme.fg_muted)
                welcome.append(f"{self._llm_provider}\n", style=theme.accent)
                if self._llm_model:
                    welcome.append(f"Model: ", style=theme.fg_muted)
                    welcome.append(f"{self._llm_model}\n", style=theme.fg_base)
            else:
                welcome.append("⚠️ No LLM provider configured. Using simulation mode.\n", style=theme.warning)
                welcome.append("Configure in Settings → AI Settings.\n", style=theme.fg_muted)
            
            welcome.append("\n" + "━" * 50 + "\n", style=theme.border)
            welcome.append("Press F1 for keyboard shortcuts\n", style=theme.fg_subtle)
            
            chat_log.write(welcome)
        except Exception:
            pass
    
    def compose_main(self):
        """Compose the main content."""
        with Horizontal(classes="main-container"):
            # Chat area (left side)
            with Vertical(classes="chat-area"):
                # Header
                with Container(classes="header-section"):
                    yield Static("🤖 Proxima AI Assistant", classes="header-title")
                    yield Static("Your quantum computing copilot", classes="header-subtitle")
                
                # Chat log
                with ScrollableContainer(classes="chat-log-container"):
                    yield RichLog(
                        auto_scroll=True,
                        classes="chat-log",
                        id="chat-log",
                        markup=True,
                        highlight=True,
                    )
                
                # Input section
                with Vertical(classes="input-section"):
                    with Horizontal(classes="input-container"):
                        yield TextArea(
                            id="prompt-input",
                            classes="prompt-input",
                        )
                        yield Button(
                            "Send ↵",
                            id="btn-send",
                            classes="send-btn",
                            variant="primary",
                        )
                    
                    with Horizontal(classes="controls-row"):
                        yield Button("🛑 Stop", id="btn-stop", classes="control-btn", variant="error", disabled=True)
                        yield Button("🗑️ Clear", id="btn-clear", classes="control-btn")
                        yield Button("📥 Import", id="btn-import", classes="control-btn")
                        yield Button("📤 Export", id="btn-export", classes="control-btn")
                        yield Button("🆕 New Chat", id="btn-new", classes="control-btn", variant="success")
                    
                    yield Static(
                        "Enter = Send | Shift+Enter = New Line | Ctrl+J/L = History | F1 = Shortcuts",
                        classes="input-hint"
                    )
            
            # Sidebar (right side)
            with Vertical(classes="sidebar-panel"):
                # Stats section
                with Container(classes="sidebar-section"):
                    yield Static("📊 Session Statistics", classes="sidebar-title")
                    with Vertical(classes="stats-grid"):
                        with Horizontal(classes="stat-row"):
                            yield Static("Provider:", classes="stat-label")
                            yield Static("—", classes="stat-value", id="stat-provider")
                        with Horizontal(classes="stat-row"):
                            yield Static("Model:", classes="stat-label")
                            yield Static("—", classes="stat-value", id="stat-model")
                        with Horizontal(classes="stat-row"):
                            yield Static("Messages:", classes="stat-label")
                            yield Static("0", classes="stat-value stat-highlight", id="stat-messages")
                        with Horizontal(classes="stat-row"):
                            yield Static("Tokens:", classes="stat-label")
                            yield Static("0", classes="stat-value", id="stat-tokens")
                        with Horizontal(classes="stat-row"):
                            yield Static("Requests:", classes="stat-label")
                            yield Static("0", classes="stat-value", id="stat-requests")
                        with Horizontal(classes="stat-row"):
                            yield Static("Avg Time:", classes="stat-label")
                            yield Static("0ms", classes="stat-value", id="stat-avg-time")
                        with Horizontal(classes="stat-row"):
                            yield Static("Session:", classes="stat-label")
                            yield Static("0m", classes="stat-value", id="stat-session-time")
                
                # Shortcuts section
                with Container(classes="sidebar-section shortcuts-panel"):
                    yield Static("⌨️ Keyboard Shortcuts", classes="sidebar-title")
                    with Vertical(classes="shortcuts-list"):
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Enter", classes="shortcut-key")
                            yield Static("Send message", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Shift+↵", classes="shortcut-key")
                            yield Static("New line", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+J", classes="shortcut-key")
                            yield Static("Previous prompt", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+L", classes="shortcut-key")
                            yield Static("Next prompt", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+N", classes="shortcut-key")
                            yield Static("New chat", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+S", classes="shortcut-key")
                            yield Static("Export chat", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+O", classes="shortcut-key")
                            yield Static("Import chat", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+Z", classes="shortcut-key")
                            yield Static("Undo", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Ctrl+Y", classes="shortcut-key")
                            yield Static("Redo", classes="shortcut-desc")
                        with Horizontal(classes="shortcut-row"):
                            yield Static("Esc", classes="shortcut-key")
                            yield Static("Go back", classes="shortcut-desc")
                
                # Recent chats section
                with Container(classes="sidebar-section"):
                    yield Static("📜 Recent Chats", classes="sidebar-title")
                
                with ScrollableContainer(classes="history-list", id="history-list"):
                    pass  # Will be populated dynamically
    
    def _update_stats_display(self) -> None:
        """Update the statistics display."""
        try:
            # Provider
            provider = self._llm_provider or 'none'
            provider_names = {
                'none': 'Not configured',
                'local': 'Ollama (Local)',
                'ollama': 'Ollama (Local)',
                'openai': 'OpenAI',
                'anthropic': 'Anthropic',
                'google': 'Google AI',
                'xai': 'xAI (Grok)',
                'deepseek': 'DeepSeek',
                'mistral': 'Mistral AI',
                'groq': 'Groq',
                'together': 'Together AI',
            }
            self.query_one("#stat-provider", Static).update(
                provider_names.get(provider, provider.title())
            )
            
            # Model
            model = self._llm_model or '—'
            if len(model) > 20:
                model = model[:17] + "..."
            self.query_one("#stat-model", Static).update(model)
            
            # Messages
            self.query_one("#stat-messages", Static).update(
                str(len(self._current_session.messages))
            )
            
            # Tokens
            self.query_one("#stat-tokens", Static).update(
                str(self._current_session.total_tokens)
            )
            
            # Requests
            self.query_one("#stat-requests", Static).update(
                str(self._current_session.total_requests)
            )
            
            # Average time
            if self._stats['total_requests'] > 0:
                avg_time = self._stats['avg_response_time'] / self._stats['total_requests']
                self.query_one("#stat-avg-time", Static).update(f"{avg_time:.0f}ms")
            
            # Session time
            elapsed = int(time.time() - self._stats['session_start'])
            minutes = elapsed // 60
            self.query_one("#stat-session-time", Static).update(f"{minutes}m")
            
        except Exception:
            pass
    
    def _update_history_list(self) -> None:
        """Update the recent chats list."""
        try:
            container = self.query_one("#history-list", ScrollableContainer)
            
            # Clear existing
            for child in list(container.children):
                child.remove()
            
            # Add recent sessions
            for session in self._sessions[:5]:
                name = session.name[:20] + "..." if len(session.name) > 20 else session.name
                msg_count = len(session.messages)
                
                item_text = f"{name}\n{msg_count} messages"
                container.mount(Static(item_text, classes="history-item"))
        except Exception:
            pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        btn_id = event.button.id
        
        if btn_id == "btn-send":
            self._send_message()
        elif btn_id == "btn-stop":
            self._stop_generation()
        elif btn_id == "btn-clear":
            self.action_clear_chat()
        elif btn_id == "btn-import":
            self.action_import_chat()
        elif btn_id == "btn-export":
            self.action_export_chat()
        elif btn_id == "btn-new":
            self.action_new_chat()
    
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text area changes for undo tracking."""
        if event.text_area.id == "prompt-input":
            # Track for undo
            current = event.text_area.text
            if self._undo_stack and self._undo_stack[-1] == current:
                return
            self._undo_stack.append(current)
            if len(self._undo_stack) > 50:
                self._undo_stack.pop(0)
            self._redo_stack.clear()
    
    def on_key(self, event) -> None:
        """Handle key events - Enter sends, Shift+Enter new line."""
        # Only handle Enter key
        if event.key != "enter":
            return
        
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
            if not input_widget.has_focus:
                return
            
            # Shift+Enter = new line (let it pass through)
            if event.shift:
                return
            
            # Plain Enter = send message
            event.prevent_default()
            event.stop()
            self._send_message()
        except Exception:
            pass
    
    def _send_message(self) -> None:
        """Send the current message."""
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
            message = input_widget.text.strip()
            
            if not message:
                return
            
            # Clear input
            input_widget.text = ""
            
            # Add to history
            if message not in self._prompt_history:
                self._prompt_history.append(message)
            self._prompt_history_index = -1
            
            # Show user message
            self._show_user_message(message)
            
            # Add to session
            self._current_session.messages.append(
                ChatMessage(role='user', content=message)
            )
            
            # Generate response
            self._is_generating = True
            self.query_one("#btn-stop", Button).disabled = False
            self.query_one("#btn-send", Button).disabled = True
            
            self._generate_response(message)
            
        except Exception as e:
            self._show_error(str(e))
    
    def _show_user_message(self, message: str) -> None:
        """Display user message in chat."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", RichLog)
            
            text = Text()
            text.append("\n👤 You\n", style=f"bold {theme.primary}")
            text.append(message + "\n", style=theme.fg_base)
            
            chat_log.write(text)
        except Exception:
            pass
    
    def _show_ai_message(self, message: str) -> None:
        """Display AI message in chat."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", RichLog)
            
            text = Text()
            text.append("\n🤖 AI\n", style=f"bold {theme.accent}")
            text.append(message + "\n", style=theme.fg_base)
            
            chat_log.write(text)
        except Exception:
            pass
    
    def _show_error(self, error: str) -> None:
        """Display error in chat."""
        try:
            theme = get_theme()
            chat_log = self.query_one("#chat-log", RichLog)
            
            text = Text()
            text.append("\n❌ Error: ", style=f"bold {theme.error}")
            text.append(error + "\n", style=theme.fg_muted)
            
            chat_log.write(text)
        except Exception:
            pass
    
    def _generate_response(self, message: str) -> None:
        """Generate AI response."""
        start_time = time.time()
        
        # Provider mapping - map TUI names to LLM router provider names
        # All major providers are now supported
        provider_map = {
            'local': 'ollama', 'ollama': 'ollama',
            'openai': 'openai', 'anthropic': 'anthropic',
            'google': 'google', 'gemini': 'google',  # Google Gemini
            'xai': 'xai', 'grok': 'xai',              # xAI Grok
            'deepseek': 'deepseek', 'mistral': 'mistral',
            'groq': 'groq', 'together': 'together',
            'openrouter': 'openrouter', 'cohere': 'cohere',
            'lmstudio': 'lmstudio', 'llamacpp': 'llama_cpp',
            'perplexity': 'perplexity', 'fireworks': 'fireworks',
            'huggingface': 'huggingface', 'replicate': 'replicate',
            'azure_openai': 'azure_openai', 'azure': 'azure_openai',
        }
        # Only a few providers still need simulation (no direct API support)
        simulation_providers = {'vertex_ai', 'aws_bedrock', 'ai21', 'deepinfra'}
        
        # Use simulation for unsupported providers
        if self._llm_provider in simulation_providers:
            router_provider = None  # Will trigger simulation
        else:
            router_provider = provider_map.get(self._llm_provider, self._llm_provider)
        
        llm_success = False
        
        if router_provider and router_provider != 'none' and self._llm_router and LLM_AVAILABLE:
            try:
                # Build context from recent messages
                context = "\n".join([
                    f"{'User' if m.role == 'user' else 'AI'}: {m.content}"
                    for m in self._current_session.messages[-5:]
                ])
                
                request = LLMRequest(
                    prompt=f"{message}",
                    system_prompt=f"""You are a helpful AI assistant for the Proxima quantum computing platform.
You help users with quantum circuit design, backend configuration, performance optimization, and understanding quantum computing concepts.
Be concise, accurate, and helpful. Format code with proper syntax.

Recent conversation context:
{context}""",
                    temperature=0.7,
                    max_tokens=1024,
                    provider=router_provider,
                    model=self._llm_model if self._llm_model else None,
                )
                
                response = self._llm_router.route(request)
                
                # Check for valid response
                response_text = ""
                if response:
                    if hasattr(response, 'text') and response.text:
                        response_text = response.text.strip()
                    
                    # Check for error
                    if hasattr(response, 'error') and response.error:
                        # Show error but continue with simulation
                        self._show_error(f"LLM Error: {response.error}")
                
                if response_text:
                    self._show_ai_message(response_text)
                    
                    # Update stats
                    elapsed = int((time.time() - start_time) * 1000)
                    tokens = getattr(response, 'tokens_used', 0) or getattr(response, 'total_tokens', 0) or 0
                    
                    self._current_session.messages.append(
                        ChatMessage(
                            role='assistant',
                            content=response_text,
                            tokens=tokens,
                            thinking_time_ms=elapsed
                        )
                    )
                    self._current_session.total_tokens += tokens
                    self._current_session.total_requests += 1
                    self._stats['total_requests'] += 1
                    self._stats['avg_response_time'] += elapsed
                    
                    self._update_stats_display()
                    self._save_current_session()
                    
                    self._finish_generation()
                    llm_success = True
                    
            except Exception as e:
                # Show what went wrong
                error_msg = str(e)
                if '404' not in error_msg and 'connection' not in error_msg.lower():
                    self._show_error(f"LLM Error: {error_msg}")
        
        if not llm_success:
            # Fallback to intelligent simulation
            self._simulate_response(message, start_time)
    
    def _simulate_response(self, message: str, start_time: float) -> None:
        """Generate simulated response."""
        msg_lower = message.lower()
        
        if any(word in msg_lower for word in ['hello', 'hi', 'hey', 'greet']):
            response = "Hello! I'm your AI assistant for Proxima. I can help you with quantum computing tasks, backend configuration, and understanding quantum concepts. What would you like to know?"
        
        elif any(word in msg_lower for word in ['backend', 'configure', 'setup']):
            response = """For backend configuration, here are the key steps:

1. **Choose a backend type:**
   - CPU Simulator: Good for small circuits (<20 qubits)
   - GPU Accelerated: Required for large circuits (>20 qubits)

2. **Configure settings:**
   - Set max qubits based on your hardware
   - Enable GPU if available (CUDA required)
   - Choose precision (float32 for speed, float64 for accuracy)

3. **Test the connection:**
   - Use the "Test Connection" button
   - Check logs for any errors

Would you like more details on any specific aspect?"""
        
        elif any(word in msg_lower for word in ['circuit', 'quantum', 'gate']):
            response = """Quantum circuits in Proxima can be created using various gates:

**Single-qubit gates:**
- H (Hadamard): Creates superposition
- X, Y, Z (Pauli): Bit/phase flips
- T, S: Phase gates

**Two-qubit gates:**
- CNOT: Controlled-NOT for entanglement
- CZ: Controlled-Z
- SWAP: Swaps qubit states

**Example Bell State:**
```
H(0)  # Create superposition
CNOT(0, 1)  # Entangle qubits
```

What specific circuit would you like to create?"""
        
        elif any(word in msg_lower for word in ['help', 'what', 'how', 'can you']):
            response = """I can help you with:

• **Circuit Design**: Creating and optimizing quantum circuits
• **Backend Selection**: Choosing the right simulator for your task
• **Performance**: Optimizing execution speed and memory usage
• **Troubleshooting**: Diagnosing common issues
• **Concepts**: Explaining quantum computing fundamentals

Just ask your question and I'll do my best to help!"""
        
        else:
            response = """I understand you're asking about quantum computing. Here are some things I can help with:

• Quantum circuit design and optimization
• Backend configuration (CPU, GPU, cloud)
• Performance tuning and benchmarking
• Understanding quantum algorithms

Could you provide more details about what you'd like to accomplish?"""
        
        # Show response
        self._show_ai_message(response)
        
        # Update stats
        elapsed = int((time.time() - start_time) * 1000)
        
        self._current_session.messages.append(
            ChatMessage(
                role='assistant',
                content=response,
                thinking_time_ms=elapsed
            )
        )
        self._current_session.total_requests += 1
        self._stats['total_requests'] += 1
        self._stats['avg_response_time'] += elapsed
        
        self._update_stats_display()
        self._save_current_session()
        self._finish_generation()
    
    def _finish_generation(self) -> None:
        """Finish generation and reset UI."""
        self._is_generating = False
        try:
            self.query_one("#btn-stop", Button).disabled = True
            self.query_one("#btn-send", Button).disabled = False
        except Exception:
            pass
    
    def _stop_generation(self) -> None:
        """Stop current generation."""
        self._is_generating = False
        self._finish_generation()
        self.notify("Generation stopped", severity="warning")
    
    # Action handlers
    def action_undo(self) -> None:
        """Undo last input change."""
        try:
            if len(self._undo_stack) > 1:
                current = self._undo_stack.pop()
                self._redo_stack.append(current)
                input_widget = self.query_one("#prompt-input", TextArea)
                input_widget.text = self._undo_stack[-1] if self._undo_stack else ""
        except Exception:
            pass
    
    def action_redo(self) -> None:
        """Redo last undone change."""
        try:
            if self._redo_stack:
                text = self._redo_stack.pop()
                self._undo_stack.append(text)
                input_widget = self.query_one("#prompt-input", TextArea)
                input_widget.text = text
        except Exception:
            pass
    
    def action_prev_prompt(self) -> None:
        """Navigate to previous prompt in history."""
        if not self._prompt_history:
            return
        
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
            
            if self._prompt_history_index == -1:
                self._prompt_history_index = len(self._prompt_history) - 1
            elif self._prompt_history_index > 0:
                self._prompt_history_index -= 1
            
            if 0 <= self._prompt_history_index < len(self._prompt_history):
                input_widget.text = self._prompt_history[self._prompt_history_index]
        except Exception:
            pass
    
    def action_next_prompt(self) -> None:
        """Navigate to next prompt in history."""
        if not self._prompt_history:
            return
        
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
            
            if self._prompt_history_index < len(self._prompt_history) - 1:
                self._prompt_history_index += 1
                input_widget.text = self._prompt_history[self._prompt_history_index]
            else:
                self._prompt_history_index = -1
                input_widget.text = ""
        except Exception:
            pass
    
    def action_new_chat(self) -> None:
        """Start a new chat session."""
        # Save current session if it has messages
        if self._current_session.messages:
            self._save_current_session()
            self._sessions.insert(0, self._current_session)
        
        # Create new session
        self._current_session = ChatSession(
            provider=self._llm_provider,
            model=self._llm_model,
        )
        
        # Clear chat log
        try:
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.clear()
        except Exception:
            pass
        
        self._show_welcome_message()
        self._update_stats_display()
        self._update_history_list()
        self.notify("New chat started", severity="success")
    
    def action_clear_chat(self) -> None:
        """Clear current chat."""
        try:
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.clear()
            self._current_session.messages.clear()
            self._show_welcome_message()
            self._update_stats_display()
            self.notify("Chat cleared", severity="information")
        except Exception:
            pass
    
    def action_export_chat(self) -> None:
        """Export current chat to file."""
        try:
            export_dir = Path("exports")
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = export_dir / f"ai_chat_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(self._current_session.to_dict(), f, indent=2)
            
            self.notify(f"Chat exported to {filename}", severity="success")
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")
    
    def action_import_chat(self) -> None:
        """Import chat from file."""
        try:
            # Look for recent exports
            export_dir = Path("exports")
            if not export_dir.exists():
                self.notify("No exports directory found", severity="warning")
                return
            
            # Find most recent chat export
            chat_files = list(export_dir.glob("ai_chat_*.json"))
            if not chat_files:
                self.notify("No chat exports found", severity="warning")
                return
            
            latest = max(chat_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest, 'r') as f:
                data = json.load(f)
            
            # Load session
            self._current_session = ChatSession.from_dict(data)
            
            # Restore chat log
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.clear()
            
            self._show_welcome_message()
            
            # Replay messages
            theme = get_theme()
            for msg in self._current_session.messages:
                text = Text()
                if msg.role == 'user':
                    text.append("\n👤 You\n", style=f"bold {theme.primary}")
                else:
                    text.append("\n🤖 AI\n", style=f"bold {theme.accent}")
                text.append(msg.content + "\n", style=theme.fg_base)
                chat_log.write(text)
            
            self._update_stats_display()
            self.notify(f"Chat imported from {latest.name}", severity="success")
            
        except Exception as e:
            self.notify(f"Import failed: {e}", severity="error")
    
    def action_show_shortcuts(self) -> None:
        """Show keyboard shortcuts help."""
        self.notify(
            "Enter=Send | Shift+Enter=New Line | Ctrl+J/L=History | Ctrl+N=New | Ctrl+S=Export | Ctrl+O=Import",
            severity="information",
            timeout=5
        )
    
    def action_copy_or_cancel(self) -> None:
        """Handle Ctrl+C - copy or cancel generation."""
        if self._is_generating:
            self._stop_generation()
    
    def action_select_all(self) -> None:
        """Select all text in input."""
        try:
            input_widget = self.query_one("#prompt-input", TextArea)
            input_widget.select_all()
        except Exception:
            pass
    
    def action_goto_ai_assistant(self) -> None:
        """Already on AI Assistant screen."""
        pass
