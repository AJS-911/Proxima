"""Markdown rendering configuration for Proxima TUI.

Custom markdown styling for help screens and documentation.
"""

from rich.markdown import Markdown
from rich.console import Console
from rich.theme import Theme
from .theme import get_theme


def create_markdown_theme() -> Theme:
    """Create a Rich theme for markdown rendering."""
    t = get_theme()
    return Theme({
        "markdown.h1": f"bold {t.primary}",
        "markdown.h2": f"bold {t.secondary}",
        "markdown.h3": f"bold {t.accent}",
        "markdown.h4": f"bold {t.fg_base}",
        "markdown.code": f"bold {t.accent} on {t.bg_subtle}",
        "markdown.code_block": f"{t.fg_base} on {t.bg_subtle}",
        "markdown.link": f"underline {t.primary}",
        "markdown.link_url": f"{t.fg_muted}",
        "markdown.block_quote": f"italic {t.fg_muted}",
        "markdown.list.bullet": f"{t.primary}",
        "markdown.list.number": f"{t.primary}",
        "markdown.item": f"{t.fg_base}",
        "markdown.hr": f"{t.border}",
        "markdown.em": f"italic {t.fg_base}",
        "markdown.strong": f"bold {t.fg_base}",
    })


def render_markdown(text: str, width: int = 80) -> str:
    """Render markdown text with Proxima styling.
    
    Args:
        text: Markdown text to render
        width: Maximum width
    
    Returns:
        Rendered text
    """
    console = Console(
        theme=create_markdown_theme(),
        width=width,
        force_terminal=True,
    )
    md = Markdown(text)
    with console.capture() as capture:
        console.print(md)
    return capture.get()


# Common markdown snippets for the TUI
HELP_HEADER = """
# Proxima TUI Help

Welcome to the Proxima Quantum Simulation Orchestration Framework.
"""

KEYBOARD_HELP = """
## Keyboard Navigation

| Key | Action |
|-----|--------|
| `1-5` | Switch screens |
| `Ctrl+P` | Command palette |
| `Ctrl+Q` | Quit |
| `?` | Show help |
| `Tab` | Next focus |
| `Esc` | Back/Cancel |
"""

EXECUTION_HELP = """
## Execution Controls

| Key | Action |
|-----|--------|
| `P` | Pause execution |
| `R` | Resume execution |
| `A` | Abort execution |
| `Z` | Rollback to checkpoint |
| `L` | Toggle log panel |
"""

COMMAND_HELP = """
## Command Palette

Press `Ctrl+P` to open the command palette.

- Use `Tab` to switch between categories
- Type to filter commands
- Press `Enter` to execute
- Press `Esc` to cancel
"""
