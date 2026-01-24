"""Common dialog utilities for Proxima TUI.

Shared utilities and base components for dialogs.
"""

from typing import Optional, List, Callable, Any

from textual.screen import ModalScreen
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Button
from rich.text import Text

from ..styles.theme import get_theme


class DialogButton:
    """Configuration for a dialog button."""
    
    def __init__(
        self,
        label: str,
        id: str,
        variant: str = "default",
        keybinding: Optional[str] = None,
        disabled: bool = False,
    ):
        """Initialize the button config.
        
        Args:
            label: Button label
            id: Button ID
            variant: Button variant (default, primary, success, warning, error)
            keybinding: Optional keyboard shortcut
            disabled: Whether button is disabled
        """
        self.label = label
        self.id = id
        self.variant = variant
        self.keybinding = keybinding
        self.disabled = disabled
    
    def get_display_label(self) -> str:
        """Get the display label with keybinding."""
        if self.keybinding:
            return f"[{self.keybinding}] {self.label}"
        return self.label


def create_button_row(
    buttons: List[DialogButton],
    classes: str = "dialog-buttons",
) -> Horizontal:
    """Create a horizontal row of buttons.
    
    Args:
        buttons: List of button configurations
        classes: CSS classes for the container
        
    Returns:
        Horizontal container with buttons
    """
    container = Horizontal(classes=classes)
    for btn in buttons:
        button = Button(
            btn.get_display_label(),
            id=btn.id,
            variant=btn.variant,
            disabled=btn.disabled,
        )
        button.classes = "dialog-button"
        container.mount(button)
    return container


def render_keybindings_help(bindings: List[tuple]) -> Text:
    """Render keybinding help text.
    
    Args:
        bindings: List of (key, description) tuples
        
    Returns:
        Rich Text with formatted keybindings
    """
    theme = get_theme()
    text = Text()
    
    for i, (key, description) in enumerate(bindings):
        if i > 0:
            text.append(" │ ", style=theme.border)
        text.append(key, style=f"bold {theme.accent}")
        text.append(f" {description}", style=theme.fg_muted)
    
    return text


class DialogHeader(Static):
    """Standard dialog header."""
    
    DEFAULT_CSS = """
    DialogHeader {
        width: 100%;
        height: auto;
        padding: 1;
        text-align: center;
        border-bottom: solid $primary-darken-2;
    }
    """
    
    def __init__(self, title: str, icon: Optional[str] = None, **kwargs):
        """Initialize the header.
        
        Args:
            title: Dialog title
            icon: Optional icon to display
        """
        super().__init__(**kwargs)
        self._title = title
        self.icon = icon
    
    def render(self) -> Text:
        """Render the header."""
        theme = get_theme()
        text = Text()
        
        if self.icon:
            text.append(f"{self.icon} ", style=theme.primary)
        
        text.append(self._title, style=f"bold {theme.primary}")
        
        return text


class DialogFooter(Static):
    """Standard dialog footer with keybindings."""
    
    DEFAULT_CSS = """
    DialogFooter {
        width: 100%;
        height: auto;
        padding: 1;
        text-align: center;
        border-top: solid $primary-darken-2;
    }
    """
    
    def __init__(self, bindings: List[tuple], **kwargs):
        """Initialize the footer.
        
        Args:
            bindings: List of (key, description) tuples
        """
        super().__init__(**kwargs)
        self.bindings = bindings
    
    def render(self) -> Text:
        """Render the footer."""
        return render_keybindings_help(self.bindings)


class FilterableList(Static):
    """Base class for filterable list dialogs."""
    
    def __init__(
        self,
        items: List[Any],
        filter_fn: Optional[Callable[[Any, str], bool]] = None,
        **kwargs,
    ):
        """Initialize the filterable list.
        
        Args:
            items: List of items
            filter_fn: Function to filter items by query
        """
        super().__init__(**kwargs)
        self.items = items
        self.filtered_items = items.copy()
        self.filter_fn = filter_fn or self._default_filter
        self.selected_index = 0
    
    def _default_filter(self, item: Any, query: str) -> bool:
        """Default filter function."""
        return query.lower() in str(item).lower()
    
    def filter(self, query: str) -> None:
        """Filter items by query."""
        if not query:
            self.filtered_items = self.items.copy()
        else:
            self.filtered_items = [
                item for item in self.items
                if self.filter_fn(item, query)
            ]
        self.selected_index = 0
        self.refresh()
    
    def select_next(self) -> None:
        """Select next item."""
        if self.selected_index < len(self.filtered_items) - 1:
            self.selected_index += 1
            self.refresh()
    
    def select_previous(self) -> None:
        """Select previous item."""
        if self.selected_index > 0:
            self.selected_index -= 1
            self.refresh()
    
    def get_selected(self) -> Optional[Any]:
        """Get currently selected item."""
        if 0 <= self.selected_index < len(self.filtered_items):
            return self.filtered_items[self.selected_index]
        return None
