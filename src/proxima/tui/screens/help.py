"""Help screen for Proxima TUI.

Help documentation and keyboard shortcuts.
"""

from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.widgets import Static, Markdown
from rich.text import Text

from .base import BaseScreen
from ..styles.theme import get_theme


HELP_CONTENT = """
# Proxima TUI Help

Welcome to the Proxima Quantum Simulation Orchestration Framework.

## Navigation

| Key | Action |
|-----|--------|
| `1` | Go to Dashboard |
| `2` | Go to Execution Monitor |
| `3` | Go to Results Browser |
| `4` | Go to Backend Management |
| `5` | Go to Settings |
| `?` | Show this Help |
| `Ctrl+P` | Open Command Palette |
| `Ctrl+Q` | Quit Application |
| `Esc` | Go Back / Cancel |
| `Tab` | Next Focus |

## Execution Controls

| Key | Action |
|-----|--------|
| `P` | Pause Execution |
| `R` | Resume Execution |
| `A` | Abort Execution |
| `Z` | Rollback to Checkpoint |
| `L` | Toggle Log Panel |

## Command Palette

Press `Ctrl+P` to open the command palette. You can:

- Search for commands by typing
- Use `Tab` to switch between categories
- Press `Enter` to execute a command
- Press `Esc` to cancel

### Available Command Categories

1. **Execution** - Run, pause, resume, abort simulations
2. **Session** - Create, switch, export sessions
3. **Backend** - Switch backend, run health checks
4. **LLM** - Configure language model settings

## Quick Start

1. From the Dashboard, click "Run Simulation" or press `Ctrl+P`
2. Select a circuit/task and backend
3. Monitor execution on the Execution screen
4. View results in the Results Browser

## Backends

Proxima supports multiple quantum simulation backends:

- **LRET** - Local Realistic Entanglement Theory
- **Cirq** - Google's quantum framework
- **Qiskit Aer** - IBM's quantum simulator
- **cuQuantum** - NVIDIA GPU acceleration
- **qsim** - High-performance simulator
- **QuEST** - Quantum Exact Simulation Toolkit

## Getting More Help

- Documentation: https://proxima.dev/docs
- GitHub: https://github.com/AJS-911/Proxima
- Report Issues: https://github.com/AJS-911/Proxima/issues
"""


class HelpScreen(BaseScreen):
    """Help and documentation screen.
    
    Shows:
    - Keyboard shortcuts
    - Quick start guide
    - Feature documentation
    """
    
    SCREEN_NAME = "help"
    SCREEN_TITLE = "Help & Documentation"
    SHOW_SIDEBAR = False
    
    DEFAULT_CSS = """
    HelpScreen .help-container {
        width: 100%;
        height: 100%;
        padding: 1 2;
    }
    
    HelpScreen .help-content {
        width: 100%;
        height: 1fr;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface;
    }
    """
    
    def compose_main(self):
        """Compose the help screen content."""
        with Vertical(classes="main-content help-container"):
            with ScrollableContainer(classes="help-content"):
                yield Markdown(HELP_CONTENT)
