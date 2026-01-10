"""Step 6.1: Terminal UI - Textual-based TUI for Proxima.

Screens:
1. Dashboard   - System status, recent executions
2. Execution   - Real-time progress, logs
3. Configuration - Settings management
4. Results     - Browse and analyze results
5. Backends    - Backend status and management

Design Principles:
- Keyboard-first navigation
- Responsive to terminal size
- Consistent color theme
- Contextual help (press ? for help)
"""

from .app import ProximaApp

# Controllers
from .controllers import (
    DataController,
    EventBus,
    EventType,
    ExecutionController,
    ExecutionStatus,
    NavigationController,
    State,
    StateStore,
    TUIEvent,
)

# Modals
from .modals import (
    ChoiceModal,
    ConfirmModal,
    ConsentModal,
    DialogResult,
    ErrorModal,
    FormField,
    FormModal,
    InputModal,
    ModalResponse,
    ProgressModal,
    show_confirm,
    show_consent,
    show_error,
    show_input,
)

# Screens
from .screens import (
    BackendsScreen,
    ConfigurationScreen,
    DashboardScreen,
    ExecutionScreen,
    ResultsScreen,
)

# Styles
from .styles import (
    ColorPalette,
    StyleManager,
    Theme,
    build_theme_css,
    get_css,
    get_palette,
    set_theme,
)

# Widgets
from .widgets import (
    BackendCard,
    BackendInfo,
    BackendStatus,
    ConfigInput,
    ConfigToggle,
    ExecutionCard,
    ExecutionProgress,
    ExecutionTimer,
    HelpModal,
    LogEntry,
    LogViewer,
    MetricDisplay,
    ProgressBar,
    ResultsTable,
    StatusIndicator,
    StatusItem,
    StatusLevel,
    StatusPanel,
)

__all__ = [
    # Main app
    "ProximaApp",
    # Screens
    "DashboardScreen",
    "ExecutionScreen",
    "ConfigurationScreen",
    "ResultsScreen",
    "BackendsScreen",
    # Widgets
    "StatusPanel",
    "StatusIndicator",
    "StatusItem",
    "StatusLevel",
    "LogViewer",
    "LogEntry",
    "ProgressBar",
    "BackendCard",
    "BackendInfo",
    "BackendStatus",
    "ResultsTable",
    "HelpModal",
    "ExecutionTimer",
    "MetricDisplay",
    "ExecutionProgress",
    "ConfigInput",
    "ConfigToggle",
    "ExecutionCard",
    # Controllers
    "EventType",
    "TUIEvent",
    "EventBus",
    "State",
    "StateStore",
    "NavigationController",
    "DataController",
    "ExecutionController",
    "ExecutionStatus",
    # Modals
    "DialogResult",
    "ModalResponse",
    "ConfirmModal",
    "InputModal",
    "ChoiceModal",
    "ProgressModal",
    "ErrorModal",
    "ConsentModal",
    "FormField",
    "FormModal",
    "show_confirm",
    "show_input",
    "show_error",
    "show_consent",
    # Styles
    "Theme",
    "ColorPalette",
    "StyleManager",
    "get_palette",
    "build_theme_css",
    "get_css",
    "set_theme",
]
