"""Code viewer component for Proxima TUI.

Syntax-highlighted code display.
"""

from typing import Optional

from textual.widgets import Static
from textual.containers import Vertical, ScrollableContainer
from rich.text import Text
from rich.syntax import Syntax

from ...styles.theme import get_theme


class CodeViewer(Vertical):
    """Code viewer with syntax highlighting.
    
    Features:
    - Syntax highlighting for multiple languages
    - Line numbers
    - Horizontal scrolling
    - Search highlighting
    """
    
    DEFAULT_CSS = """
    CodeViewer {
        height: 100%;
        border: solid $primary-darken-2;
        background: $surface-darken-1;
    }
    
    CodeViewer .code-header {
        height: auto;
        padding: 1;
        border-bottom: solid $primary-darken-3;
        background: $surface;
    }
    
    CodeViewer .code-content {
        height: 1fr;
        padding: 1;
    }
    
    CodeViewer .code-footer {
        height: auto;
        padding: 0 1;
        border-top: solid $primary-darken-3;
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        code: str = "",
        language: str = "python",
        title: Optional[str] = None,
        show_line_numbers: bool = True,
        theme: str = "monokai",
        **kwargs,
    ):
        """Initialize the code viewer.
        
        Args:
            code: The code to display
            language: Programming language for syntax highlighting
            title: Optional title for the viewer
            show_line_numbers: Whether to show line numbers
            theme: Syntax highlighting theme
        """
        super().__init__(**kwargs)
        self.code = code
        self.language = language
        self._title = title or f"{language.capitalize()} Code"
        self.show_line_numbers = show_line_numbers
        self.syntax_theme = theme
    
    def compose(self):
        """Compose the code viewer."""
        yield CodeHeader(
            self._title,
            self.language,
            len(self.code.splitlines()),
            classes="code-header",
        )
        with ScrollableContainer(classes="code-content"):
            yield CodeContent(
                self.code,
                self.language,
                self.show_line_numbers,
                self.syntax_theme,
            )
        yield CodeFooter(
            len(self.code.splitlines()),
            len(self.code),
            classes="code-footer",
        )
    
    def set_code(self, code: str, language: Optional[str] = None) -> None:
        """Set the code to display.
        
        Args:
            code: The code to display
            language: Optional language override
        """
        self.code = code
        if language:
            self.language = language
        
        # Update content
        content = self.query_one(CodeContent)
        content.code = code
        content.language = self.language
        content.refresh()
        
        # Update stats
        self.query_one(CodeFooter).lines = len(code.splitlines())
        self.query_one(CodeFooter).chars = len(code)
        self.query_one(CodeFooter).refresh()
        
        self.query_one(CodeHeader).lines = len(code.splitlines())
        self.query_one(CodeHeader).refresh()


class CodeHeader(Static):
    """Header for the code viewer."""
    
    def __init__(
        self,
        title: str,
        language: str,
        lines: int,
        **kwargs,
    ):
        """Initialize the header."""
        super().__init__(**kwargs)
        self._title = title
        self.language = language
        self.lines = lines
    
    def render(self) -> Text:
        """Render the header."""
        theme = get_theme()
        text = Text()
        
        text.append(self._title, style=f"bold {theme.primary}")
        text.append("  │  ", style=theme.border)
        text.append(self.language, style=theme.accent)
        text.append(f"  │  {self.lines} lines", style=theme.fg_muted)
        
        return text


class CodeContent(Static):
    """Content area showing the code."""
    
    def __init__(
        self,
        code: str,
        language: str,
        show_line_numbers: bool = True,
        syntax_theme: str = "monokai",
        **kwargs,
    ):
        """Initialize the content."""
        super().__init__(**kwargs)
        self.code = code
        self.language = language
        self.show_line_numbers = show_line_numbers
        self.syntax_theme = syntax_theme
    
    def render(self):
        """Render the code with syntax highlighting."""
        if not self.code:
            theme = get_theme()
            return Text("No code to display", style=theme.fg_muted)
        
        return Syntax(
            self.code,
            self.language,
            theme=self.syntax_theme,
            line_numbers=self.show_line_numbers,
            word_wrap=False,
        )


class CodeFooter(Static):
    """Footer with code statistics."""
    
    def __init__(self, lines: int = 0, chars: int = 0, **kwargs):
        """Initialize the footer."""
        super().__init__(**kwargs)
        self.lines = lines
        self.chars = chars
    
    def render(self) -> Text:
        """Render the footer."""
        theme = get_theme()
        text = Text()
        
        text.append(f"{self.lines} lines", style=theme.fg_muted)
        text.append("  │  ", style=theme.border)
        text.append(f"{self.chars} characters", style=theme.fg_muted)
        
        return text
