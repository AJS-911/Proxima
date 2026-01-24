"""Main ProximaTUI Application.

The main Textual application class for the Proxima TUI.
"""

from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen

from .state import TUIState
from .styles.theme import get_theme

# CSS file path
CSS_PATH = Path(__file__).parent / "styles" / "base.tcss"


class ProximaTUI(App):
    """Proxima Terminal User Interface.
    
    A professional, Crush-inspired TUI for quantum simulation orchestration.
    """
    
    TITLE = "Proxima"
    SUB_TITLE = "Quantum Simulation Orchestration"
    
    CSS_PATH = CSS_PATH
    
    BINDINGS = [
        Binding("1", "goto_dashboard", "Dashboard", show=True),
        Binding("2", "goto_execution", "Execution", show=True),
        Binding("3", "goto_results", "Results", show=True),
        Binding("4", "goto_backends", "Backends", show=True),
        Binding("5", "goto_settings", "Settings", show=True),
        Binding("ctrl+p", "open_commands", "Commands", show=True),
        Binding("question_mark", "show_help", "Help", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]
    
    def __init__(
        self,
        theme: str = "dark",
        initial_screen: str = "dashboard",
    ):
        """Initialize the ProximaTUI application.
        
        Args:
            theme: Theme name ('dark' or 'light')
            initial_screen: Initial screen to show
        """
        super().__init__()
        self.theme_name = theme
        self.initial_screen_name = initial_screen
        self.state = TUIState()
        self._theme = get_theme()
    
    def on_mount(self) -> None:
        """Handle application mount."""
        from .screens import DashboardScreen
        
        self.title = "Proxima"
        self.sub_title = "Quantum Simulation Orchestration"
        
        # Install and push the initial screen
        self.push_screen(DashboardScreen(state=self.state))
    
    def _navigate_to_screen(self, screen_name: str) -> None:
        """Navigate to a specific screen.
        
        Args:
            screen_name: Name of the screen to navigate to
        """
        from .screens import (
            DashboardScreen,
            ExecutionScreen,
            ResultsScreen,
            BackendsScreen,
            SettingsScreen,
            HelpScreen,
        )
        
        screens = {
            "dashboard": DashboardScreen,
            "execution": ExecutionScreen,
            "results": ResultsScreen,
            "backends": BackendsScreen,
            "settings": SettingsScreen,
            "help": HelpScreen,
        }
        
        screen_class = screens.get(screen_name)
        if screen_class:
            self.state.current_screen = screen_name
            self.push_screen(screen_class(state=self.state))
    
    def action_goto_dashboard(self) -> None:
        """Navigate to dashboard screen."""
        self._navigate_to_screen("dashboard")
    
    def action_goto_execution(self) -> None:
        """Navigate to execution screen."""
        self._navigate_to_screen("execution")
    
    def action_goto_results(self) -> None:
        """Navigate to results screen."""
        self._navigate_to_screen("results")
    
    def action_goto_backends(self) -> None:
        """Navigate to backends screen."""
        self._navigate_to_screen("backends")
    
    def action_goto_settings(self) -> None:
        """Navigate to settings screen."""
        self._navigate_to_screen("settings")
    
    def action_show_help(self) -> None:
        """Show help screen."""
        self._navigate_to_screen("help")
    
    def action_open_commands(self) -> None:
        """Open command palette."""
        from .dialogs import CommandPalette
        
        def handle_command(command):
            if command:
                self.notify(f"Executing: {command.name}")
                # Execute command action if defined
                if command.action:
                    command.action()
        
        self.push_screen(CommandPalette(), handle_command)
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


def launch(theme: str = "dark", initial_screen: str = "dashboard") -> None:
    """Launch the Proxima TUI.
    
    Args:
        theme: Theme name ('dark' or 'light')
        initial_screen: Initial screen to show
    """
    app = ProximaTUI(theme=theme, initial_screen=initial_screen)
    app.run()


# Backward compatibility alias
ProximaApp = ProximaTUI


if __name__ == "__main__":
    launch()
