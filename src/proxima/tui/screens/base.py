"""Base screen class for Proxima TUI.

Provides common functionality for all screens.
"""

from typing import Optional

from textual.screen import Screen
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header

from ..state import TUIState
from ..styles.theme import get_theme
from ..components.sidebar import Sidebar


class BaseScreen(Screen):
    """Base screen with common layout and functionality.
    
    All screens inherit from this to get:
    - Consistent layout with sidebar
    - Footer with keybindings
    - Access to TUI state
    - Common styling
    """
    
    DEFAULT_CSS = """
    BaseScreen {
        layout: horizontal;
    }
    
    BaseScreen > .main-area {
        width: 1fr;
        height: 100%;
    }
    
    BaseScreen > .main-content {
        width: 100%;
        height: 1fr;
        padding: 1 2;
    }
    
    BaseScreen > Footer {
        dock: bottom;
    }
    """
    
    BINDINGS = [
        ("1", "goto_dashboard", "Dashboard"),
        ("2", "goto_execution", "Execution"),
        ("3", "goto_results", "Results"),
        ("4", "goto_backends", "Backends"),
        ("5", "goto_settings", "Settings"),
        ("question_mark", "show_help", "Help"),
        ("ctrl+p", "open_commands", "Commands"),
        ("escape", "go_back", "Back"),
    ]
    
    # Override in subclasses
    SCREEN_NAME = "base"
    SCREEN_TITLE = "Proxima"
    SHOW_SIDEBAR = True
    
    def __init__(
        self,
        state: Optional[TUIState] = None,
        **kwargs,
    ):
        """Initialize the base screen.
        
        Args:
            state: TUI state (uses app state if None)
        """
        super().__init__(**kwargs)
        self._state = state
    
    @property
    def state(self) -> TUIState:
        """Get the TUI state."""
        if self._state:
            return self._state
        # Try to get from app
        if hasattr(self.app, "state"):
            return self.app.state
        return TUIState()
    
    def compose(self):
        """Compose the screen layout."""
        with Horizontal():
            # Main content area
            with Vertical(classes="main-area"):
                yield from self.compose_main()
            
            # Sidebar (if enabled)
            if self.SHOW_SIDEBAR:
                yield Sidebar(self.state)
        
        yield Footer()
    
    def compose_main(self):
        """Compose the main content area.
        
        Override in subclasses to provide screen-specific content.
        """
        yield Vertical(classes="main-content")
    
    def on_mount(self) -> None:
        """Handle screen mount."""
        self.title = self.SCREEN_TITLE
    
    def action_goto_dashboard(self) -> None:
        """Navigate to dashboard."""
        self.app.action_goto_dashboard()
    
    def action_goto_execution(self) -> None:
        """Navigate to execution."""
        self.app.action_goto_execution()
    
    def action_goto_results(self) -> None:
        """Navigate to results."""
        self.app.action_goto_results()
    
    def action_goto_backends(self) -> None:
        """Navigate to backends."""
        self.app.action_goto_backends()
    
    def action_goto_settings(self) -> None:
        """Navigate to settings."""
        self.app.action_goto_settings()
    
    def action_show_help(self) -> None:
        """Show help."""
        self.app.action_show_help()
    
    def action_open_commands(self) -> None:
        """Open command palette."""
        self.app.action_open_commands()
    
    def action_go_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
