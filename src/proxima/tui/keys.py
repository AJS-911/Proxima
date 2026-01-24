"""Global keybinding definitions for Proxima TUI.

Centralized key mappings for consistent keyboard navigation.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class KeyBinding:
    """A keyboard binding definition."""
    
    key: str
    action: str
    description: str
    context: str = "global"  # global, execution, dialog, etc.


# Global Navigation Keys
GLOBAL_KEYS: List[KeyBinding] = [
    KeyBinding("1", "goto_dashboard", "Go to Dashboard"),
    KeyBinding("2", "goto_execution", "Go to Execution"),
    KeyBinding("3", "goto_results", "Go to Results"),
    KeyBinding("4", "goto_backends", "Go to Backends"),
    KeyBinding("5", "goto_settings", "Go to Settings"),
    KeyBinding("?", "show_help", "Show Help"),
    KeyBinding("ctrl+p", "open_commands", "Open Command Palette"),
    KeyBinding("ctrl+q", "quit", "Quit Application"),
    KeyBinding("ctrl+c", "cancel", "Cancel/Abort"),
    KeyBinding("escape", "back", "Cancel/Back"),
    KeyBinding("tab", "next_focus", "Next Focus"),
    KeyBinding("shift+tab", "prev_focus", "Previous Focus"),
]

# Execution Screen Keys
EXECUTION_KEYS: List[KeyBinding] = [
    KeyBinding("p", "pause", "Pause Execution", "execution"),
    KeyBinding("r", "resume", "Resume Execution", "execution"),
    KeyBinding("a", "abort", "Abort Execution", "execution"),
    KeyBinding("z", "rollback", "Rollback to Checkpoint", "execution"),
    KeyBinding("l", "toggle_log", "Toggle Log Panel", "execution"),
]

# Dialog Keys
DIALOG_KEYS: List[KeyBinding] = [
    KeyBinding("enter", "confirm", "Confirm/Select", "dialog"),
    KeyBinding("escape", "cancel", "Cancel/Close", "dialog"),
    KeyBinding("up", "navigate_up", "Navigate Up", "dialog"),
    KeyBinding("down", "navigate_down", "Navigate Down", "dialog"),
    KeyBinding("tab", "switch_category", "Switch Category", "dialog"),
]

# Permission Dialog Keys
PERMISSION_KEYS: List[KeyBinding] = [
    KeyBinding("a", "allow", "Allow", "permission"),
    KeyBinding("s", "allow_session", "Allow for Session", "permission"),
    KeyBinding("d", "deny", "Deny", "permission"),
    KeyBinding("t", "toggle_diff", "Toggle Diff View", "permission"),
]


def get_keybindings_for_context(context: str) -> List[KeyBinding]:
    """Get all keybindings for a specific context."""
    all_keys = GLOBAL_KEYS + EXECUTION_KEYS + DIALOG_KEYS + PERMISSION_KEYS
    return [k for k in all_keys if k.context == context or k.context == "global"]


def get_help_text(context: str = "global") -> str:
    """Generate help text for keybindings in a context."""
    keys = get_keybindings_for_context(context)
    lines = []
    for key in keys:
        lines.append(f"{key.key:<12} {key.description}")
    return "\n".join(lines)
