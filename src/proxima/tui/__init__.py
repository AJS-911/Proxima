"""Proxima TUI - Terminal User Interface.

A professional, Crush-inspired Terminal User Interface for the Proxima
Quantum Simulation Orchestration Framework.

Features:
- Professional dark theme with magenta/purple accents
- Right-aligned sidebar with real-time status
- Command palette with fuzzy search
- Permission dialogs for consent management
- Real-time execution monitoring
- Quantum-specific visualizations
"""

from .app import ProximaTUI, launch
from .state import TUIState
from .screens import (
    BaseScreen,
    DashboardScreen,
    ExecutionScreen,
    ResultsScreen,
    BackendsScreen,
    SettingsScreen,
    HelpScreen,
)
from .dialogs import (
    CommandPalette,
    PermissionsDialog,
    ConfirmationDialog,
    InputDialog,
    ModelsDialog,
    BackendsDialog,
    SessionsDialog,
    ErrorDialog,
)
from . import util

# Backward compatibility aliases
from .styles_compat import Theme, ColorPalette, get_palette, get_css
from .modals import ConfirmModal, DialogResult, ModalResponse
from .widgets import StatusPanel, LogPanel, BackendCard, ExecutionCard, ProgressBar
from .controllers import EventBus, StateManager, NavigationController

# Alias for backward compatibility
ProximaApp = ProximaTUI

__all__ = [
    # Main app
    "ProximaTUI",
    "ProximaApp",  # Backward compatibility alias
    "launch",
    "TUIState",
    # Screens
    "BaseScreen",
    "DashboardScreen",
    "ExecutionScreen",
    "ResultsScreen",
    "BackendsScreen",
    "SettingsScreen",
    "HelpScreen",
    # Dialogs
    "CommandPalette",
    "PermissionsDialog",
    "ConfirmationDialog",
    "InputDialog",
    "ModelsDialog",
    "BackendsDialog",
    "SessionsDialog",
    "ErrorDialog",
    # Backward compatibility - styles
    "Theme",
    "ColorPalette",
    "get_palette",
    "get_css",
    # Backward compatibility - modals
    "ConfirmModal",
    "DialogResult",
    "ModalResponse",
    # Backward compatibility - widgets
    "StatusPanel",
    "LogPanel",
    "BackendCard",
    "ExecutionCard",
    "ProgressBar",
    # Backward compatibility - controllers
    "EventBus",
    "StateManager",
    "NavigationController",
    # Utilities
    "util",
]
