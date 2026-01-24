"""Backward compatibility shim for proxima.tui.styles.

This module provides backward compatibility with the old TUI architecture.
New code should use proxima.tui.styles.theme instead.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional


class Theme(Enum):
    """Theme enumeration for backward compatibility."""
    DARK = "dark"
    LIGHT = "light"
    OCEAN = "ocean"
    FOREST = "forest"
    SUNSET = "sunset"


@dataclass
class ColorPalette:
    """Color palette for backward compatibility."""
    primary: str = "#FF00FF"
    secondary: str = "#AA00FF"
    background: str = "#121212"
    surface: str = "#1a1a1a"
    error: str = "#FF3333"
    warning: str = "#FFAA00"
    success: str = "#00FF66"
    info: str = "#00AAFF"
    text: str = "#FFFFFF"
    text_muted: str = "#B0B0B0"
    border: str = "#333333"
    
    def to_css_vars(self) -> str:
        """Convert palette to CSS variables."""
        return f"""
        $primary: {self.primary};
        $secondary: {self.secondary};
        $background: {self.background};
        $surface: {self.surface};
        $error: {self.error};
        $warning: {self.warning};
        $success: {self.success};
        $info: {self.info};
        $text: {self.text};
        $text-muted: {self.text_muted};
        $border: {self.border};
        """


# Predefined palettes
DARK_PALETTE = ColorPalette()
LIGHT_PALETTE = ColorPalette(
    background="#FFFFFF",
    surface="#F5F5F5",
    text="#000000",
    text_muted="#666666",
    border="#CCCCCC",
)


def get_palette(theme: Theme = Theme.DARK) -> ColorPalette:
    """Get color palette for theme."""
    if theme == Theme.LIGHT:
        return LIGHT_PALETTE
    return DARK_PALETTE


def build_theme_css(palette: ColorPalette) -> str:
    """Build CSS from palette."""
    return f"""
    Screen {{
        background: {palette.background};
        color: {palette.text};
    }}
    
    .panel {{
        background: {palette.surface};
        border: solid {palette.border};
    }}
    
    Button {{
        background: {palette.primary};
    }}
    
    Button:hover {{
        background: {palette.secondary};
    }}
    """


def get_css() -> str:
    """Get default CSS."""
    return build_theme_css(DARK_PALETTE)


__all__ = [
    "Theme",
    "ColorPalette",
    "DARK_PALETTE",
    "LIGHT_PALETTE",
    "get_palette",
    "build_theme_css",
    "get_css",
]
