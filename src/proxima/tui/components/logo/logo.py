"""ASCII PROXIMA Logo with gradient coloring.

A dynamic logo component inspired by Crush's logo implementation.
"""

import random
from typing import List, Optional

from textual.widget import Widget
from textual.widgets import Static
from rich.text import Text

from ...styles.theme import get_theme, create_gradient, apply_gradient_to_text


# ASCII letterforms for PROXIMA
LETTER_P = [
    "█████╗ ",
    "██╔══██╗",
    "█████╔╝",
    "██╔═══╝ ",
    "██║    ",
    "╚═╝    ",
]

LETTER_R = [
    "██████╗ ",
    "██╔══██╗",
    "██████╔╝",
    "██╔══██╗",
    "██║  ██║",
    "╚═╝  ╚═╝",
]

LETTER_O = [
    " ██████╗ ",
    "██╔═══██╗",
    "██║   ██║",
    "██║   ██║",
    "╚██████╔╝",
    " ╚═════╝ ",
]

LETTER_X = [
    "██╗  ██╗",
    "╚██╗██╔╝",
    " ╚███╔╝ ",
    " ██╔██╗ ",
    "██╔╝ ██╗",
    "╚═╝  ╚═╝",
]

LETTER_I = [
    "██╗",
    "██║",
    "██║",
    "██║",
    "██║",
    "╚═╝",
]

LETTER_M = [
    "███╗   ███╗",
    "████╗ ████║",
    "██╔████╔██║",
    "██║╚██╔╝██║",
    "██║ ╚═╝ ██║",
    "╚═╝     ╚═╝",
]

LETTER_A = [
    " █████╗ ",
    "██╔══██╗",
    "███████║",
    "██╔══██║",
    "██║  ██║",
    "╚═╝  ╚═╝",
]

LETTERFORMS = [LETTER_P, LETTER_R, LETTER_O, LETTER_X, LETTER_I, LETTER_M, LETTER_A]

# Simpler ASCII logo for normal use
SIMPLE_LOGO = [
    "╔═╗╦═╗╔═╗╦ ╦╦╔╦╗╔═╗",
    "╠═╝╠╦╝║ ║╔╩╦╝║║║║╠═╣",
    "╩  ╩╚═╚═╝╩ ╚═╩╩ ╩╩ ╩",
]

# Compact logo
COMPACT_LOGO = "◈ PROXIMA"


class Logo(Static):
    """ASCII PROXIMA logo with gradient coloring.
    
    Features:
    - Gradient coloring from secondary (purple) to primary (magenta)
    - Stretchable letters (random letter stretches on each render)
    - Compact mode for smaller terminals
    - Diagonal field lines on sides
    """
    
    DEFAULT_CSS = """
    Logo {
        width: 100%;
        height: auto;
        content-align: center middle;
    }
    
    Logo.-compact {
        height: 1;
    }
    """
    
    def __init__(
        self,
        compact: bool = False,
        show_version: bool = False,
        version: str = "",
        animate_stretch: bool = False,
        **kwargs,
    ):
        """Initialize the logo.
        
        Args:
            compact: Use compact single-line logo
            show_version: Show version number
            version: Version string
            animate_stretch: Randomly stretch letters
        """
        super().__init__(**kwargs)
        self.compact = compact
        self.show_version = show_version
        self.version = version
        self.animate_stretch = animate_stretch
        self._stretch_index = -1
    
    def render(self) -> Text:
        """Render the logo."""
        if self.compact:
            return self._render_compact()
        return self._render_full()
    
    def _render_compact(self) -> Text:
        """Render compact single-line logo."""
        theme = get_theme()
        
        text = Text()
        text.append("◈ ", style=f"bold {theme.secondary}")
        
        # Apply gradient to PROXIMA text
        gradient = create_gradient(
            [theme.secondary, theme.primary, theme.primary_light],
            len("PROXIMA"),
        )
        
        for i, char in enumerate("PROXIMA"):
            text.append(char, style=f"bold {gradient[i]}")
        
        if self.show_version and self.version:
            text.append(f" v{self.version}", style=f"dim {theme.fg_muted}")
        
        return text
    
    def _render_full(self) -> Text:
        """Render full ASCII logo."""
        theme = get_theme()
        
        # Use simple logo for better compatibility
        logo_lines = SIMPLE_LOGO.copy()
        
        # Create gradient for the logo
        max_width = max(len(line) for line in logo_lines)
        gradient = create_gradient(
            [theme.secondary, theme.primary, theme.primary_light],
            max_width,
        )
        
        text = Text()
        
        for line_idx, line in enumerate(logo_lines):
            for char_idx, char in enumerate(line):
                if char.strip():
                    color = gradient[min(char_idx, len(gradient) - 1)]
                    text.append(char, style=f"bold {color}")
                else:
                    text.append(char)
            
            if line_idx < len(logo_lines) - 1:
                text.append("\n")
        
        # Add version below logo
        if self.show_version and self.version:
            text.append("\n")
            version_text = f"v{self.version}"
            # Center the version
            padding = (max_width - len(version_text)) // 2
            text.append(" " * padding)
            text.append(version_text, style=f"dim {theme.fg_muted}")
        
        return text
    
    def _render_with_letterforms(self) -> Text:
        """Render logo using individual letterforms (advanced).
        
        This creates a more stylized logo with optional letter stretching.
        """
        theme = get_theme()
        
        # Optionally stretch a random letter
        if self.animate_stretch:
            self._stretch_index = random.randint(0, len(LETTERFORMS) - 1)
        
        # Combine letterforms
        num_lines = len(LETTER_P)
        lines = []
        
        for line_idx in range(num_lines):
            line_parts = []
            for letter_idx, letterform in enumerate(LETTERFORMS):
                letter_line = letterform[line_idx]
                
                # Apply stretch effect
                if letter_idx == self._stretch_index:
                    # Double a character in the middle
                    mid = len(letter_line) // 2
                    letter_line = letter_line[:mid] + letter_line[mid] + letter_line[mid:]
                
                line_parts.append(letter_line)
            
            lines.append("".join(line_parts))
        
        # Create gradient
        max_width = max(len(line) for line in lines)
        gradient = create_gradient(
            [theme.secondary, theme.primary, theme.primary_light],
            max_width,
        )
        
        text = Text()
        
        for line_idx, line in enumerate(lines):
            for char_idx, char in enumerate(line):
                if char.strip():
                    color = gradient[min(char_idx, len(gradient) - 1)]
                    text.append(char, style=f"bold {color}")
                else:
                    text.append(char)
            
            if line_idx < len(lines) - 1:
                text.append("\n")
        
        return text
    
    def refresh_stretch(self) -> None:
        """Refresh the letter stretch effect."""
        if self.animate_stretch:
            self._stretch_index = random.randint(0, len(LETTERFORMS) - 1)
            self.refresh()
