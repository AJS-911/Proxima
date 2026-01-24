"""Section component for sidebar and panels.

Provides consistent section headers and collapsible sections.
"""

from textual.widget import Widget
from textual.widgets import Static
from textual.containers import Vertical
from textual.reactive import reactive
from rich.text import Text

from ...styles.theme import get_theme


class SectionHeader(Static):
    """A styled section header with optional line."""
    
    DEFAULT_CSS = """
    SectionHeader {
        height: 1;
        width: 100%;
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        title: str,
        show_line: bool = True,
        line_width: int = 20,
        **kwargs,
    ):
        """Initialize the section header.
        
        Args:
            title: Section title
            show_line: Whether to show the horizontal line
            line_width: Width of the line
        """
        super().__init__(**kwargs)
        self.title = title
        self.show_line = show_line
        self.line_width = line_width
    
    def render(self) -> Text:
        """Render the section header."""
        theme = get_theme()
        
        text = Text()
        text.append(self.title, style=f"bold {theme.fg_subtle}")
        
        if self.show_line:
            # Calculate remaining space for line
            remaining = max(0, self.line_width - len(self.title) - 1)
            text.append(" ")
            text.append("─" * remaining, style=theme.border)
        
        return text


class Section(Vertical):
    """A collapsible section container.
    
    Provides a consistent layout for sidebar sections with
    header, content, and collapse functionality.
    """
    
    DEFAULT_CSS = """
    Section {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }
    
    Section > .section-content {
        padding-left: 1;
    }
    
    Section.-collapsed > .section-content {
        display: none;
    }
    """
    
    collapsed = reactive(False)
    
    def __init__(
        self,
        title: str,
        *children,
        collapsible: bool = True,
        initially_collapsed: bool = False,
        **kwargs,
    ):
        """Initialize the section.
        
        Args:
            title: Section title
            *children: Child widgets
            collapsible: Whether the section can be collapsed
            initially_collapsed: Whether to start collapsed
        """
        super().__init__(**kwargs)
        self.title = title
        self.collapsible = collapsible
        self.collapsed = initially_collapsed
        self._children = children
    
    def compose(self):
        """Compose the section layout."""
        yield SectionHeader(self.title)
        with Vertical(classes="section-content"):
            for child in self._children:
                yield child
    
    def toggle_collapse(self) -> None:
        """Toggle the collapsed state."""
        if self.collapsible:
            self.collapsed = not self.collapsed
            self.set_class(self.collapsed, "-collapsed")
    
    def expand(self) -> None:
        """Expand the section."""
        if self.collapsed:
            self.toggle_collapse()
    
    def collapse(self) -> None:
        """Collapse the section."""
        if not self.collapsed:
            self.toggle_collapse()
