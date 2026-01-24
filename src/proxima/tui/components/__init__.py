"""Proxima TUI Components Package.

Contains all UI components and widgets.
"""

from .logo import Logo
from .sidebar import Sidebar
from .progress import ProgressBar, StageTimeline
from .editor import InputArea, AutocompleteInput
from .viewers import LogViewer, ResultViewer, DiffViewer, CodeViewer

__all__ = [
    "Logo",
    "Sidebar",
    "ProgressBar",
    "StageTimeline",
    "InputArea",
    "AutocompleteInput",
    "LogViewer",
    "ResultViewer",
    "DiffViewer",
    "CodeViewer",
]
