"""Layout utilities for Proxima TUI.

Provides layout calculation and sizing utilities.
"""

from typing import Protocol, Optional
from dataclasses import dataclass


class Sizeable(Protocol):
    """Protocol for components that report their size."""
    
    def get_min_height(self) -> int:
        """Get minimum height needed."""
        ...
    
    def get_preferred_height(self) -> int:
        """Get preferred height."""
        ...
    
    def get_max_height(self) -> int:
        """Get maximum height."""
        ...


@dataclass
class SizeConstraints:
    """Size constraints for a component."""
    
    min_height: int = 0
    preferred_height: int = 0
    max_height: int = -1  # -1 means unlimited
    min_width: int = 0
    preferred_width: int = 0
    max_width: int = -1


def calculate_height(
    available: int,
    components: list[tuple[str, SizeConstraints]],
    priorities: Optional[dict[str, int]] = None,
) -> dict[str, int]:
    """Calculate heights for multiple components.
    
    Distributes available height among components based on their
    constraints and priorities.
    
    Args:
        available: Total available height
        components: List of (name, constraints) tuples
        priorities: Optional dict of name -> priority (higher = more space)
    
    Returns:
        Dict of component name to allocated height
    """
    if not components:
        return {}
    
    priorities = priorities or {}
    
    # Start with minimum heights
    allocated = {name: constraints.min_height for name, constraints in components}
    
    # Calculate remaining space
    used = sum(allocated.values())
    remaining = available - used
    
    if remaining <= 0:
        return allocated
    
    # Sort by priority (higher first)
    sorted_components = sorted(
        components,
        key=lambda x: priorities.get(x[0], 0),
        reverse=True,
    )
    
    # Distribute remaining space
    for name, constraints in sorted_components:
        if remaining <= 0:
            break
        
        current = allocated[name]
        preferred = constraints.preferred_height
        max_height = constraints.max_height if constraints.max_height > 0 else available
        
        # Try to reach preferred height
        if current < preferred:
            add = min(preferred - current, remaining, max_height - current)
            allocated[name] += add
            remaining -= add
    
    # If still remaining, give to highest priority with room
    if remaining > 0:
        for name, constraints in sorted_components:
            if remaining <= 0:
                break
            
            current = allocated[name]
            max_height = constraints.max_height if constraints.max_height > 0 else available
            
            if current < max_height:
                add = min(max_height - current, remaining)
                allocated[name] += add
                remaining -= add
    
    return allocated


def calculate_sidebar_sections(
    available_height: int,
    has_logo: bool = True,
    compact_mode: bool = False,
    has_session: bool = False,
    has_execution: bool = False,
) -> dict[str, int]:
    """Calculate heights for sidebar sections.
    
    Args:
        available_height: Total available height
        has_logo: Whether to show logo
        compact_mode: Whether in compact mode
        has_session: Whether there's an active session
        has_execution: Whether there's an active execution
    
    Returns:
        Dict of section name to allocated height
    """
    used = 0
    
    # Fixed heights
    if has_logo:
        used += 7 if not compact_mode else 2
    used += 1  # Empty line after logo
    
    if has_session:
        used += 2  # Task title + empty
        if not compact_mode:
            used += 2  # CWD + empty
    
    # Model info
    used += 3
    
    # Section headers (3 sections × 2 lines each = 6)
    used += 6
    
    # Padding
    used += 2
    
    remaining = max(0, available_height - used)
    
    # Distribute remaining among dynamic sections
    sections = [
        ("results", SizeConstraints(min_height=2, preferred_height=5, max_height=10)),
        ("backends", SizeConstraints(min_height=2, preferred_height=4, max_height=8)),
        ("memory", SizeConstraints(min_height=2, preferred_height=3, max_height=3)),
    ]
    
    if has_execution:
        sections.append(
            ("checkpoints", SizeConstraints(min_height=2, preferred_height=3, max_height=4))
        )
    
    return calculate_height(
        remaining,
        sections,
        priorities={"results": 3, "backends": 2, "memory": 1, "checkpoints": 4},
    )
