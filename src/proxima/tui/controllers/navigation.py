"""Navigation Controller for Proxima TUI.

Handles screen navigation and history management.
"""

from typing import List, Optional, Callable
from dataclasses import dataclass, field

from ..state import TUIState
from ..state.events import ScreenChanged


@dataclass
class NavigationEntry:
    """A navigation history entry."""
    screen: str
    data: Optional[dict] = None


class NavigationController:
    """Controller for screen navigation.
    
    Manages screen transitions, navigation history, and screen state.
    """
    
    VALID_SCREENS = [
        "dashboard",
        "execution",
        "results",
        "backends",
        "settings",
        "help",
    ]
    
    def __init__(self, state: TUIState):
        """Initialize the navigation controller.
        
        Args:
            state: The TUI state instance
        """
        self.state = state
        self._history: List[NavigationEntry] = []
        self._forward_stack: List[NavigationEntry] = []
        self._max_history = 50
        self._on_navigate_callbacks: List[Callable] = []
    
    @property
    def current_screen(self) -> str:
        """Get the current screen name."""
        return self.state.current_screen
    
    @property
    def can_go_back(self) -> bool:
        """Check if we can navigate back."""
        return len(self._history) > 1
    
    @property
    def can_go_forward(self) -> bool:
        """Check if we can navigate forward."""
        return len(self._forward_stack) > 0
    
    def navigate_to(
        self,
        screen: str,
        data: Optional[dict] = None,
        add_to_history: bool = True,
    ) -> bool:
        """Navigate to a screen.
        
        Args:
            screen: Screen name to navigate to
            data: Optional data to pass to the screen
            add_to_history: Whether to add to navigation history
        
        Returns:
            True if navigation was successful
        """
        if screen not in self.VALID_SCREENS:
            return False
        
        previous_screen = self.state.current_screen
        
        # Add current to history before navigating
        if add_to_history and previous_screen:
            entry = NavigationEntry(screen=previous_screen)
            self._history.append(entry)
            
            # Clear forward stack on new navigation
            self._forward_stack.clear()
            
            # Trim history if too long
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        
        # Update state
        self.state.current_screen = screen
        
        # Notify callbacks
        for callback in self._on_navigate_callbacks:
            callback(ScreenChanged(
                previous_screen=previous_screen,
                new_screen=screen,
            ))
        
        return True
    
    def go_back(self) -> bool:
        """Navigate to the previous screen.
        
        Returns:
            True if navigation was successful
        """
        if not self.can_go_back:
            return False
        
        # Move current to forward stack
        current = NavigationEntry(screen=self.state.current_screen)
        self._forward_stack.append(current)
        
        # Pop from history
        entry = self._history.pop()
        
        return self.navigate_to(entry.screen, entry.data, add_to_history=False)
    
    def go_forward(self) -> bool:
        """Navigate forward in history.
        
        Returns:
            True if navigation was successful
        """
        if not self.can_go_forward:
            return False
        
        # Move current to history
        current = NavigationEntry(screen=self.state.current_screen)
        self._history.append(current)
        
        # Pop from forward stack
        entry = self._forward_stack.pop()
        
        return self.navigate_to(entry.screen, entry.data, add_to_history=False)
    
    def on_navigate(self, callback: Callable) -> None:
        """Register a navigation callback.
        
        Args:
            callback: Function to call on navigation
        """
        self._on_navigate_callbacks.append(callback)
    
    def get_screen_title(self, screen: Optional[str] = None) -> str:
        """Get the title for a screen.
        
        Args:
            screen: Screen name (uses current if None)
        
        Returns:
            Screen title
        """
        screen = screen or self.current_screen
        titles = {
            "dashboard": "Dashboard",
            "execution": "Execution Monitor",
            "results": "Results Browser",
            "backends": "Backend Management",
            "settings": "Configuration",
            "help": "Help & Documentation",
        }
        return titles.get(screen, screen.title())
    
    def get_screen_index(self, screen: Optional[str] = None) -> int:
        """Get the index of a screen (for navigation hints).
        
        Args:
            screen: Screen name (uses current if None)
        
        Returns:
            Screen index (1-5 for main screens, 0 for help)
        """
        screen = screen or self.current_screen
        indices = {
            "dashboard": 1,
            "execution": 2,
            "results": 3,
            "backends": 4,
            "settings": 5,
            "help": 0,
        }
        return indices.get(screen, 0)
