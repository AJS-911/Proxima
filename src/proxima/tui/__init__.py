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

__all__ = [
    "ProximaTUI",
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
    # Utilities
    "util",
]
