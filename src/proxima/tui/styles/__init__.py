"""Proxima TUI Styles Package.

Contains theme definitions, icons, and styling utilities.
"""

from .theme import ProximaTheme, get_theme
from .icons import *

# Backward compatibility imports
from ..styles_compat import (
    Theme,
    ColorPalette,
    DARK_PALETTE,
    LIGHT_PALETTE,
    get_palette,
    build_theme_css,
    get_css,
)

__all__ = [
    "ProximaTheme",
    "get_theme",
    # Backward compatibility
    "Theme",
    "ColorPalette",
    "DARK_PALETTE",
    "LIGHT_PALETTE",
    "get_palette",
    "build_theme_css",
    "get_css",
]
