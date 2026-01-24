"""Proxima TUI Utilities Package.

Utility functions and helpers.
"""

from .gradient import apply_gradient, blend_colors, interpolate_color
from .truncate import truncate_text, truncate_middle, smart_truncate
from .model import BaseModel

__all__ = [
    "apply_gradient",
    "blend_colors",
    "interpolate_color",
    "truncate_text",
    "truncate_middle",
    "smart_truncate",
    "BaseModel",
]
