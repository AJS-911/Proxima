"""Diff viewer component for Proxima TUI.

Displays diffs for comparisons between results.
"""

from dataclasses import dataclass
from typing import List, Optional

from textual.widgets import Static
from textual.containers import Vertical, ScrollableContainer
from rich.text import Text

from ...styles.theme import get_theme
from ...styles.icons import ICON_DIFF_ADD, ICON_DIFF_REMOVE, ICON_DIFF_CHANGE


@dataclass
class DiffLine:
    """A single line in a diff."""
    
    line_type: str  # 'add', 'remove', 'change', 'context'
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    line_number_old: Optional[int] = None
    line_number_new: Optional[int] = None


@dataclass
class DiffHunk:
    """A hunk of changes in a diff."""
    
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[DiffLine]


class DiffViewer(Vertical):
    """Diff viewer for comparing results or code.
    
    Features:
    - Side-by-side or unified view
    - Syntax highlighting
    - Line numbers
    - Change indicators
    """
    
    DEFAULT_CSS = """
    DiffViewer {
        height: 100%;
        border: solid $primary-darken-2;
        background: $surface;
    }
    
    DiffViewer .diff-header {
        height: auto;
        padding: 1;
        border-bottom: solid $primary-darken-3;
        background: $surface-lighten-1;
    }
    
    DiffViewer .diff-content {
        height: 1fr;
        padding: 0 1;
    }
    
    DiffViewer .diff-stats {
        height: auto;
        padding: 1;
        border-top: solid $primary-darken-3;
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        title: str = "Diff",
        old_label: str = "Before",
        new_label: str = "After",
        unified: bool = True,
        **kwargs,
    ):
        """Initialize the diff viewer.
        
        Args:
            title: Viewer title
            old_label: Label for old content
            new_label: Label for new content
            unified: Use unified diff format
        """
        super().__init__(**kwargs)
        self.title = title
        self.old_label = old_label
        self.new_label = new_label
        self.unified = unified
        self.hunks: List[DiffHunk] = []
        self.additions = 0
        self.deletions = 0
    
    def compose(self):
        """Compose the diff viewer."""
        yield DiffHeader(
            self.title,
            self.old_label,
            self.new_label,
            classes="diff-header",
        )
        with ScrollableContainer(classes="diff-content"):
            yield DiffContent(self.hunks, self.unified)
        yield DiffStats(self.additions, self.deletions, classes="diff-stats")
    
    def set_diff(self, hunks: List[DiffHunk]) -> None:
        """Set the diff to display."""
        self.hunks = hunks
        
        # Calculate stats
        self.additions = sum(
            1 for h in hunks for l in h.lines
            if l.line_type == 'add'
        )
        self.deletions = sum(
            1 for h in hunks for l in h.lines
            if l.line_type == 'remove'
        )
        
        # Update children
        self.query_one(DiffContent).hunks = hunks
        self.query_one(DiffContent).refresh()
        self.query_one(DiffStats).additions = self.additions
        self.query_one(DiffStats).deletions = self.deletions
        self.query_one(DiffStats).refresh()
    
    def set_text_diff(self, old_text: str, new_text: str) -> None:
        """Create diff from two text strings."""
        import difflib
        
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        
        differ = difflib.unified_diff(old_lines, new_lines)
        
        hunks = []
        current_hunk = None
        
        for line in differ:
            if line.startswith('@@'):
                # Parse hunk header
                if current_hunk:
                    hunks.append(current_hunk)
                current_hunk = DiffHunk(0, 0, 0, 0, [])
            elif current_hunk:
                if line.startswith('+'):
                    current_hunk.lines.append(DiffLine('add', new_content=line[1:]))
                elif line.startswith('-'):
                    current_hunk.lines.append(DiffLine('remove', old_content=line[1:]))
                else:
                    current_hunk.lines.append(DiffLine('context', old_content=line, new_content=line))
        
        if current_hunk:
            hunks.append(current_hunk)
        
        self.set_diff(hunks)


class DiffHeader(Static):
    """Header for the diff viewer."""
    
    def __init__(
        self,
        title: str,
        old_label: str,
        new_label: str,
        **kwargs,
    ):
        """Initialize the header."""
        super().__init__(**kwargs)
        self._title = title
        self.old_label = old_label
        self.new_label = new_label
    
    def render(self) -> Text:
        """Render the header."""
        theme = get_theme()
        text = Text()
        
        text.append(self._title, style=f"bold {theme.primary}")
        text.append("  │  ", style=theme.border)
        text.append(self.old_label, style=theme.diff_delete_fg)
        text.append(" → ", style=theme.fg_muted)
        text.append(self.new_label, style=theme.diff_insert_fg)
        
        return text


class DiffContent(Static):
    """Content area showing diff lines."""
    
    def __init__(self, hunks: List[DiffHunk], unified: bool = True, **kwargs):
        """Initialize the content."""
        super().__init__(**kwargs)
        self.hunks = hunks
        self.unified = unified
    
    def render(self) -> Text:
        """Render the diff content."""
        theme = get_theme()
        text = Text()
        
        if not self.hunks:
            text.append("No differences to display", style=theme.fg_muted)
            return text
        
        for hunk in self.hunks:
            for line in hunk.lines:
                if line.line_type == 'add':
                    text.append(ICON_DIFF_ADD, style=f"bold {theme.diff_insert_fg}")
                    text.append(" ")
                    content = line.new_content or ""
                    text.append(content.rstrip(), style=theme.diff_insert_fg)
                    text.append("\n")
                elif line.line_type == 'remove':
                    text.append(ICON_DIFF_REMOVE, style=f"bold {theme.diff_delete_fg}")
                    text.append(" ")
                    content = line.old_content or ""
                    text.append(content.rstrip(), style=theme.diff_delete_fg)
                    text.append("\n")
                elif line.line_type == 'change':
                    text.append(ICON_DIFF_CHANGE, style=f"bold {theme.diff_change_fg}")
                    text.append(" ")
                    content = line.new_content or ""
                    text.append(content.rstrip(), style=theme.diff_change_fg)
                    text.append("\n")
                else:  # context
                    text.append("  ")
                    content = line.old_content or ""
                    text.append(content.rstrip(), style=theme.fg_muted)
                    text.append("\n")
        
        return text


class DiffStats(Static):
    """Statistics for the diff."""
    
    def __init__(self, additions: int = 0, deletions: int = 0, **kwargs):
        """Initialize the stats."""
        super().__init__(**kwargs)
        self.additions = additions
        self.deletions = deletions
    
    def render(self) -> Text:
        """Render the stats."""
        theme = get_theme()
        text = Text()
        
        text.append(f"+{self.additions}", style=f"bold {theme.diff_insert_fg}")
        text.append(" additions  ", style=theme.fg_muted)
        text.append(f"-{self.deletions}", style=f"bold {theme.diff_delete_fg}")
        text.append(" deletions", style=theme.fg_muted)
        
        return text
