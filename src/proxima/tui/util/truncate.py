"""Text truncation utilities for Proxima TUI.

Functions for truncating text to fit within constraints.
"""

from typing import Optional


def truncate_text(text: str, max_length: int, suffix: str = "…") -> str:
    """Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if not text or max_length <= 0:
        return ""
    
    if len(text) <= max_length:
        return text
    
    if max_length <= len(suffix):
        return suffix[:max_length]
    
    return text[:max_length - len(suffix)] + suffix


def truncate_middle(text: str, max_length: int, separator: str = "…") -> str:
    """Truncate text in the middle.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including separator
        separator: Separator to use in middle
        
    Returns:
        Truncated text with middle removed
    """
    if not text or max_length <= 0:
        return ""
    
    if len(text) <= max_length:
        return text
    
    if max_length <= len(separator):
        return separator[:max_length]
    
    available = max_length - len(separator)
    start_len = available // 2 + available % 2
    end_len = available // 2
    
    return text[:start_len] + separator + text[-end_len:] if end_len > 0 else text[:start_len] + separator


def smart_truncate(
    text: str,
    max_length: int,
    word_boundary: bool = True,
    suffix: str = "…",
) -> str:
    """Truncate text intelligently at word boundaries.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        word_boundary: Whether to truncate at word boundaries
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if not text or max_length <= 0:
        return ""
    
    if len(text) <= max_length:
        return text
    
    if not word_boundary:
        return truncate_text(text, max_length, suffix)
    
    if max_length <= len(suffix):
        return suffix[:max_length]
    
    available = max_length - len(suffix)
    
    # Find last space within available length
    truncated = text[:available]
    last_space = truncated.rfind(' ')
    
    if last_space > 0 and last_space > available * 0.5:
        # Only use word boundary if it's not too aggressive
        return truncated[:last_space].rstrip() + suffix
    
    return truncated + suffix


def truncate_path(path: str, max_length: int, separator: str = "…") -> str:
    """Truncate a file path intelligently.
    
    Keeps the filename and truncates directory parts.
    
    Args:
        path: File path to truncate
        max_length: Maximum length
        separator: Separator for truncated parts
        
    Returns:
        Truncated path
    """
    if not path or max_length <= 0:
        return ""
    
    if len(path) <= max_length:
        return path
    
    # Split into parts
    import os
    parts = path.replace('\\', '/').split('/')
    
    if len(parts) <= 2:
        return truncate_middle(path, max_length, separator)
    
    # Keep first and last parts
    first = parts[0]
    last = parts[-1]
    
    # Check if just first/last fits
    minimal = f"{first}/{separator}/{last}"
    if len(minimal) > max_length:
        return truncate_middle(path, max_length, separator)
    
    # Try to include more parts from the end
    result_parts = [first, separator]
    remaining = max_length - len(first) - len(separator) - 2  # -2 for slashes
    
    # Add parts from end until we run out of space
    end_parts = []
    for part in reversed(parts[1:]):
        if remaining >= len(part) + 1:  # +1 for slash
            end_parts.insert(0, part)
            remaining -= len(part) + 1
        else:
            break
    
    result_parts.extend(end_parts)
    
    return '/'.join(result_parts)


def fit_text(text: str, width: int, align: str = "left", fill: str = " ") -> str:
    """Fit text to exact width with alignment.
    
    Args:
        text: Text to fit
        width: Target width
        align: Alignment ('left', 'right', 'center')
        fill: Fill character
        
    Returns:
        Text padded or truncated to width
    """
    if width <= 0:
        return ""
    
    if len(text) > width:
        return truncate_text(text, width)
    
    padding = width - len(text)
    
    if align == "right":
        return fill * padding + text
    elif align == "center":
        left_pad = padding // 2
        right_pad = padding - left_pad
        return fill * left_pad + text + fill * right_pad
    else:  # left
        return text + fill * padding
