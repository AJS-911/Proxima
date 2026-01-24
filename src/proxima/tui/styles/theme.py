"""Proxima TUI Theme - Quantum-inspired dark theme with magenta accents.

A comprehensive color palette and styling system inspired by Crush AI.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import colorsys


@dataclass
class ProximaTheme:
    """Proxima TUI Theme - Quantum-inspired dark theme with magenta accents."""
    
    name: str = "proxima-dark"
    is_dark: bool = True
    
    # Primary Accent Colors (Magenta/Purple gradient)
    primary: str = "#FF00FF"          # Magenta
    primary_light: str = "#FF66FF"    # Light magenta
    primary_dark: str = "#AA00AA"     # Dark magenta
    secondary: str = "#AA00FF"        # Purple
    secondary_light: str = "#CC66FF"  # Light purple
    tertiary: str = "#6600CC"         # Deep purple
    accent: str = "#00FFFF"           # Cyan (quantum highlight)
    
    # Background Colors
    bg_darkest: str = "#0a0a0a"       # Deepest background
    bg_base: str = "#121212"          # Main background
    bg_base_lighter: str = "#1a1a1a"  # Slightly lighter
    bg_subtle: str = "#242424"        # Subtle panels
    bg_overlay: str = "#2a2a2a"       # Dialogs/modals
    bg_elevated: str = "#333333"      # Elevated elements
    
    # Foreground Colors
    fg_base: str = "#FFFFFF"          # Primary text
    fg_muted: str = "#B0B0B0"         # Secondary text
    fg_half_muted: str = "#909090"    # Dimmed text
    fg_subtle: str = "#707070"        # Very dim text
    fg_selected: str = "#FFFFFF"      # Selected text
    fg_disabled: str = "#505050"      # Disabled text
    
    # Border Colors
    border: str = "#333333"           # Default border
    border_focus: str = "#FF00FF"     # Focused border
    border_subtle: str = "#2a2a2a"    # Subtle border
    
    # Status Colors
    success: str = "#00FF66"          # Success green
    success_dark: str = "#00AA44"     # Dark success
    error: str = "#FF3333"            # Error red
    error_dark: str = "#AA2222"       # Dark error
    warning: str = "#FFAA00"          # Warning orange
    warning_dark: str = "#CC8800"     # Dark warning
    info: str = "#00AAFF"             # Info blue
    info_dark: str = "#0088CC"        # Dark info
    
    # Execution State Colors
    state_idle: str = "#808080"       # Gray
    state_planning: str = "#FFAA00"   # Orange
    state_ready: str = "#00AAFF"      # Blue
    state_running: str = "#00FF66"    # Green
    state_paused: str = "#FFAA00"     # Orange
    state_completed: str = "#00FF66"  # Green
    state_error: str = "#FF3333"      # Red
    state_aborted: str = "#FF6600"    # Orange-red
    state_recovering: str = "#AA00FF" # Purple
    
    # Memory Level Colors
    memory_ok: str = "#00FF66"        # Green (< 60%)
    memory_info: str = "#00AAFF"      # Blue (60-80%)
    memory_warning: str = "#FFAA00"   # Orange (80-95%)
    memory_critical: str = "#FF6600"  # Orange-red (95-98%)
    memory_abort: str = "#FF3333"     # Red (> 98%)
    
    # Backend Health Colors
    health_healthy: str = "#00FF66"   # Green
    health_degraded: str = "#FFAA00"  # Orange
    health_unhealthy: str = "#FF3333" # Red
    health_unknown: str = "#808080"   # Gray
    
    # Diff Colors (for comparisons)
    diff_insert_bg: str = "#1a2f1a"   # Green-tinted background
    diff_insert_fg: str = "#00FF66"   # Green text
    diff_delete_bg: str = "#2f1a1a"   # Red-tinted background
    diff_delete_fg: str = "#FF6666"   # Red text
    diff_change_bg: str = "#2a2a1a"   # Yellow-tinted background
    diff_change_fg: str = "#FFFF66"   # Yellow text
    
    # Quantum-specific Colors
    qubit_zero: str = "#00AAFF"       # |0⟩ state blue
    qubit_one: str = "#FF00FF"        # |1⟩ state magenta
    entangled: str = "#AA00FF"        # Entanglement purple
    superposition: str = "#00FFFF"    # Superposition cyan
    
    # Gradient Colors (for animations)
    gradient_start: str = "#AA00FF"   # Purple
    gradient_mid: str = "#FF00FF"     # Magenta
    gradient_end: str = "#FF66FF"     # Light magenta
    
    def get_execution_state_color(self, state: str) -> str:
        """Get color for an execution state."""
        state_colors = {
            "IDLE": self.state_idle,
            "PLANNING": self.state_planning,
            "READY": self.state_ready,
            "RUNNING": self.state_running,
            "PAUSED": self.state_paused,
            "COMPLETED": self.state_completed,
            "ERROR": self.state_error,
            "ABORTED": self.state_aborted,
            "RECOVERING": self.state_recovering,
        }
        return state_colors.get(state.upper(), self.fg_muted)
    
    def get_memory_level_color(self, level: str) -> str:
        """Get color for a memory level."""
        level_colors = {
            "OK": self.memory_ok,
            "INFO": self.memory_info,
            "WARNING": self.memory_warning,
            "CRITICAL": self.memory_critical,
            "ABORT": self.memory_abort,
        }
        return level_colors.get(level.upper(), self.fg_muted)
    
    def get_health_color(self, status: str) -> str:
        """Get color for backend health status."""
        health_colors = {
            "HEALTHY": self.health_healthy,
            "DEGRADED": self.health_degraded,
            "UNHEALTHY": self.health_unhealthy,
            "UNKNOWN": self.health_unknown,
        }
        return health_colors.get(status.upper(), self.health_unknown)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex color."""
    return f"#{r:02x}{g:02x}{b:02x}"


def blend_colors(color1: str, color2: str, ratio: float = 0.5) -> str:
    """Blend two colors together.
    
    Args:
        color1: First hex color
        color2: Second hex color
        ratio: Blend ratio (0.0 = color1, 1.0 = color2)
    
    Returns:
        Blended hex color
    """
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    
    r = int(r1 + (r2 - r1) * ratio)
    g = int(g1 + (g2 - g1) * ratio)
    b = int(b1 + (b2 - b1) * ratio)
    
    return rgb_to_hex(r, g, b)


def create_gradient(colors: list[str], steps: int) -> list[str]:
    """Create a color gradient from multiple color stops.
    
    Args:
        colors: List of hex colors
        steps: Number of gradient steps
    
    Returns:
        List of hex colors forming the gradient
    """
    if len(colors) < 2:
        return colors * steps
    
    gradient = []
    segments = len(colors) - 1
    steps_per_segment = steps // segments
    
    for i in range(segments):
        for j in range(steps_per_segment):
            ratio = j / steps_per_segment
            gradient.append(blend_colors(colors[i], colors[i + 1], ratio))
    
    # Add final color
    gradient.append(colors[-1])
    
    return gradient[:steps]


def apply_gradient_to_text(text: str, colors: list[str]) -> str:
    """Apply a gradient to text using Rich markup.
    
    Args:
        text: The text to colorize
        colors: List of hex colors
    
    Returns:
        Text with Rich color markup
    """
    if not text or not colors:
        return text
    
    gradient = create_gradient(colors, len(text))
    result = []
    
    for i, char in enumerate(text):
        if char.isspace():
            result.append(char)
        else:
            color = gradient[min(i, len(gradient) - 1)]
            result.append(f"[{color}]{char}[/]")
    
    return "".join(result)


# Global theme instance
_current_theme: ProximaTheme = ProximaTheme()


def get_theme() -> ProximaTheme:
    """Get the current theme."""
    return _current_theme


def set_theme(theme: ProximaTheme) -> None:
    """Set the current theme."""
    global _current_theme
    _current_theme = theme
