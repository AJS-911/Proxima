"""Gradient color utilities for Proxima TUI.

Functions for creating color gradients and transitions.
"""

from typing import List, Tuple
import colorsys


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple.
    
    Args:
        hex_color: Hex color string (e.g., '#FF00FF')
        
    Returns:
        Tuple of (r, g, b) values 0-255
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB values to hex color.
    
    Args:
        r: Red value 0-255
        g: Green value 0-255
        b: Blue value 0-255
        
    Returns:
        Hex color string
    """
    return f"#{r:02x}{g:02x}{b:02x}"


def interpolate_color(color1: str, color2: str, factor: float) -> str:
    """Interpolate between two colors.
    
    Args:
        color1: Start color (hex)
        color2: End color (hex)
        factor: Interpolation factor (0.0 = color1, 1.0 = color2)
        
    Returns:
        Interpolated color (hex)
    """
    factor = max(0.0, min(1.0, factor))
    
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    
    r = int(r1 + (r2 - r1) * factor)
    g = int(g1 + (g2 - g1) * factor)
    b = int(b1 + (b2 - b1) * factor)
    
    return rgb_to_hex(r, g, b)


def blend_colors(size: int, *colors: str) -> List[str]:
    """Generate color ramp between multiple color stops.
    
    Args:
        size: Number of colors to generate
        *colors: Color stops (hex strings)
        
    Returns:
        List of interpolated colors
    """
    if size <= 0:
        return []
    if size == 1:
        return [colors[0]] if colors else []
    if len(colors) == 0:
        return []
    if len(colors) == 1:
        return [colors[0]] * size
    
    result = []
    segments = len(colors) - 1
    colors_per_segment = (size - 1) / segments
    
    for i in range(size):
        # Find which segment this index falls into
        segment_index = min(int(i / colors_per_segment), segments - 1)
        
        # Calculate position within segment
        segment_start = segment_index * colors_per_segment
        factor = (i - segment_start) / colors_per_segment if colors_per_segment > 0 else 0
        
        # Interpolate
        color = interpolate_color(
            colors[segment_index],
            colors[segment_index + 1],
            factor,
        )
        result.append(color)
    
    return result


def apply_gradient(text: str, color_start: str, color_end: str, bold: bool = False) -> str:
    """Apply horizontal gradient to text using Rich markup.
    
    Args:
        text: Text to colorize
        color_start: Starting color (hex)
        color_end: Ending color (hex)
        bold: Whether to make text bold
        
    Returns:
        Rich-formatted string with gradient
    """
    if not text:
        return ""
    
    colors = blend_colors(len(text), color_start, color_end)
    
    result = []
    for char, color in zip(text, colors):
        style = f"bold {color}" if bold else color
        result.append(f"[{style}]{char}[/]")
    
    return "".join(result)


def create_hue_gradient(size: int, saturation: float = 1.0, lightness: float = 0.5) -> List[str]:
    """Create a rainbow gradient through the hue spectrum.
    
    Args:
        size: Number of colors
        saturation: Color saturation (0-1)
        lightness: Color lightness (0-1)
        
    Returns:
        List of hex colors
    """
    colors = []
    for i in range(size):
        hue = i / size
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb_to_hex(int(r * 255), int(g * 255), int(b * 255)))
    return colors


def darken(color: str, factor: float = 0.2) -> str:
    """Darken a color by a factor.
    
    Args:
        color: Hex color
        factor: Darkening factor (0-1)
        
    Returns:
        Darkened hex color
    """
    r, g, b = hex_to_rgb(color)
    factor = 1 - max(0, min(1, factor))
    return rgb_to_hex(int(r * factor), int(g * factor), int(b * factor))


def lighten(color: str, factor: float = 0.2) -> str:
    """Lighten a color by a factor.
    
    Args:
        color: Hex color
        factor: Lightening factor (0-1)
        
    Returns:
        Lightened hex color
    """
    r, g, b = hex_to_rgb(color)
    factor = max(0, min(1, factor))
    return rgb_to_hex(
        int(r + (255 - r) * factor),
        int(g + (255 - g) * factor),
        int(b + (255 - b) * factor),
    )
