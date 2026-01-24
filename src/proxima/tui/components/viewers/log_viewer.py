"""Log viewer component for Proxima TUI.

Scrollable log display with filtering and search.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from textual.widgets import RichLog, Static
from textual.containers import Vertical
from rich.text import Text

from ...styles.theme import get_theme


@dataclass
class LogEntry:
    """A single log entry."""
    
    timestamp: datetime
    level: str
    message: str
    source: Optional[str] = None
    
    def render(self, theme) -> Text:
        """Render the log entry as Rich Text."""
        text = Text()
        
        # Timestamp
        time_str = self.timestamp.strftime("%H:%M:%S")
        text.append(f"[{time_str}] ", style=theme.fg_subtle)
        
        # Level
        level_colors = {
            "DEBUG": theme.fg_muted,
            "INFO": theme.info,
            "WARNING": theme.warning,
            "ERROR": theme.error,
            "CRITICAL": f"bold {theme.error}",
        }
        level_style = level_colors.get(self.level, theme.fg_muted)
        text.append(f"{self.level:<8}", style=level_style)
        
        # Source
        if self.source:
            text.append(f"[{self.source}] ", style=theme.fg_subtle)
        
        # Message
        text.append(self.message, style=theme.fg_base)
        
        return text


class LogViewer(Vertical):
    """Log viewer with filtering and controls.
    
    Features:
    - Scrollable log display
    - Level filtering
    - Search
    - Auto-scroll
    """
    
    DEFAULT_CSS = """
    LogViewer {
        height: 100%;
        border: solid $primary-darken-2;
        background: $surface-darken-1;
    }
    
    LogViewer .log-header {
        height: auto;
        padding: 0 1;
        border-bottom: solid $primary-darken-3;
        background: $surface;
    }
    
    LogViewer .log-content {
        height: 1fr;
        padding: 0 1;
    }
    
    LogViewer .log-footer {
        height: auto;
        padding: 0 1;
        border-top: solid $primary-darken-3;
        background: $surface;
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        title: str = "Log",
        auto_scroll: bool = True,
        max_entries: int = 1000,
        min_level: str = "DEBUG",
        **kwargs,
    ):
        """Initialize the log viewer.
        
        Args:
            title: Viewer title
            auto_scroll: Whether to auto-scroll to new entries
            max_entries: Maximum entries to keep
            min_level: Minimum log level to display
        """
        super().__init__(**kwargs)
        self.title = title
        self.auto_scroll = auto_scroll
        self.max_entries = max_entries
        self.min_level = min_level
        self.entries: List[LogEntry] = []
        self._level_order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    def compose(self):
        """Compose the log viewer."""
        yield Static(self.title, classes="log-header")
        yield RichLog(classes="log-content", auto_scroll=self.auto_scroll)
        yield LogFooter(classes="log-footer")
    
    def add_entry(self, entry: LogEntry) -> None:
        """Add a log entry."""
        # Check level filter
        entry_level_idx = self._level_order.index(entry.level) if entry.level in self._level_order else 0
        min_level_idx = self._level_order.index(self.min_level) if self.min_level in self._level_order else 0
        
        if entry_level_idx < min_level_idx:
            return
        
        # Add to entries
        self.entries.append(entry)
        
        # Trim if needed
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        # Render to log
        theme = get_theme()
        log = self.query_one(RichLog)
        log.write(entry.render(theme))
        
        # Update footer
        self._update_footer()
    
    def add(
        self,
        level: str,
        message: str,
        source: Optional[str] = None,
    ) -> None:
        """Add a log message.
        
        Args:
            level: Log level
            message: Log message
            source: Optional source identifier
        """
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            source=source,
        )
        self.add_entry(entry)
    
    def info(self, message: str, source: Optional[str] = None) -> None:
        """Add an info message."""
        self.add("INFO", message, source)
    
    def warning(self, message: str, source: Optional[str] = None) -> None:
        """Add a warning message."""
        self.add("WARNING", message, source)
    
    def error(self, message: str, source: Optional[str] = None) -> None:
        """Add an error message."""
        self.add("ERROR", message, source)
    
    def debug(self, message: str, source: Optional[str] = None) -> None:
        """Add a debug message."""
        self.add("DEBUG", message, source)
    
    def clear(self) -> None:
        """Clear all log entries."""
        self.entries.clear()
        self.query_one(RichLog).clear()
        self._update_footer()
    
    def set_min_level(self, level: str) -> None:
        """Set minimum log level filter."""
        self.min_level = level
        self._refresh_display()
    
    def _refresh_display(self) -> None:
        """Refresh the log display with current filter."""
        log = self.query_one(RichLog)
        log.clear()
        
        theme = get_theme()
        min_level_idx = self._level_order.index(self.min_level) if self.min_level in self._level_order else 0
        
        for entry in self.entries:
            entry_level_idx = self._level_order.index(entry.level) if entry.level in self._level_order else 0
            if entry_level_idx >= min_level_idx:
                log.write(entry.render(theme))
        
        self._update_footer()
    
    def _update_footer(self) -> None:
        """Update the footer with entry count."""
        footer = self.query_one(LogFooter)
        footer.update_count(len(self.entries), self.min_level)


class LogFooter(Static):
    """Footer for the log viewer."""
    
    def __init__(self, **kwargs):
        """Initialize the footer."""
        super().__init__(**kwargs)
        self.count = 0
        self.level = "DEBUG"
    
    def render(self) -> Text:
        """Render the footer."""
        theme = get_theme()
        text = Text()
        
        text.append(f"{self.count} entries", style=theme.fg_muted)
        text.append("  │  ", style=theme.border)
        text.append(f"Level: {self.level}", style=theme.fg_muted)
        
        return text
    
    def update_count(self, count: int, level: str) -> None:
        """Update the entry count and level."""
        self.count = count
        self.level = level
        self.refresh()
