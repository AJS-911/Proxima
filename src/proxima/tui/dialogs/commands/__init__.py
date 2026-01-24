"""Proxima TUI Commands Dialogs Package.

Command palette dialog.
"""

from .commands import CommandPalette, Command, DEFAULT_COMMANDS
from .command_item import CommandItem

__all__ = [
    "CommandPalette",
    "Command",
    "CommandItem",
    "DEFAULT_COMMANDS",
]
