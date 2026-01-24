"""Core component utilities for Proxima TUI."""

from .section import Section, SectionHeader
from .layout import Sizeable, calculate_height

__all__ = [
    "Section",
    "SectionHeader",
    "Sizeable",
    "calculate_height",
]
